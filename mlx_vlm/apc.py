"""Automatic Prefix Caching (APC) for mlx-vlm.

Hash-based, block-level KV cache reuse across requests. The KV cache is split
into fixed-size blocks (default 16 tokens). Each fully-filled block is
identified by a chained hash::

    block_hash[i] = H(block_hash[i-1], tuple(tokens[i*bs:(i+1)*bs]), extra_hash[i])

``extra_hash[i]`` carries multimodal context (e.g. an image content hash) so
identical token IDs with different images don't collide. ``H`` defaults to
Python's built-in ``hash`` (fast, deterministic within a single process). Set
``APC_HASH=sha256`` to opt into a stable cryptographic hash (~100-200 ns/tok
overhead).

Eviction is LRU with reference counting: blocks are kept alive while
``ref_cnt > 0`` and the free queue is a doubly-linked list embedded in
``APCBlock`` for O(1) move-to-tail. All blocks are pre-allocated as a pool
to avoid Python object churn. Pure in-memory; no persistence across restarts.

Numerical note: APC itself is *exact*. The K/V tensors stored in the block
pool are byte-identical to what a fresh prefill would produce — the cache
introduces no approximation, it just retains tensors. However, cold-vs-warm
runs of the same prompt can produce slightly different logits because of
**batch non-invariance** in the attention kernel: a long Q (cold prefill,
e.g. 60 tokens) and a short Q (warm-start suffix, e.g. 13 tokens against
47 cached tokens) trigger different tile shapes / reduction orders inside
flash-attention, and floating-point matmul is non-associative. The
Thinking Machines analysis (2025) and Microsoft Research's LLM-42 paper
give the formal treatment. The same drift happens without prefix caching
any time dynamic batching changes the batch composition between two
identical requests — APC just makes it visible by giving a clean
cold/warm contrast. Warm-to-warm runs *are* deterministic: identical
prompts repeated under APC always produce identical text. For
bit-equivalent cold==warm, you need batch-invariant RMSNorm / matmul /
attention kernels (vLLM's ``--enable-batch-invariance``, SGLang with
FlashInfer/FA3), not a different cache design.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger("mlx_vlm.apc")

DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_BLOCKS = 2048
SEED_PARENT_HASH = 0


def _hash_use_sha256() -> bool:
    return os.environ.get("APC_HASH", "fast").lower() == "sha256"


def _hash_tokens(parent: int, tokens: Tuple[int, ...], extra: int) -> int:
    """Chain hash for a single block."""
    if _hash_use_sha256():
        h = hashlib.sha256()
        h.update(int(parent & ((1 << 64) - 1)).to_bytes(8, "little"))
        h.update(np.asarray(tokens, dtype=np.int32).tobytes())
        h.update(int(extra & ((1 << 64) - 1)).to_bytes(8, "little"))
        return int.from_bytes(h.digest()[:8], "little", signed=True)
    return hash((parent, tokens, extra))


def hash_image_payload(
    pixel_values: Optional[mx.array] = None,
    image_ref: Any = None,
) -> int:
    """Stable content hash of an image payload.

    Prefers hashing the actual ``pixel_values`` tensor (so resize/transform
    differences invalidate the cache). Falls back to hashing the source
    identifier (path / URL / repr).
    """
    if pixel_values is not None:
        try:
            arr = np.asarray(pixel_values).astype(np.float16, copy=False)
            digest = hashlib.sha256(arr.tobytes()).digest()
            return int.from_bytes(digest[:8], "little", signed=True)
        except Exception:
            pass

    if image_ref is None:
        return 0
    if isinstance(image_ref, (list, tuple)):
        h = SEED_PARENT_HASH
        for it in image_ref:
            h = hash((h, hash_image_payload(image_ref=it)))
        return h
    if isinstance(image_ref, str):
        digest = hashlib.sha256(image_ref.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=True)
    if isinstance(image_ref, bytes):
        return int.from_bytes(
            hashlib.sha256(image_ref).digest()[:8], "little", signed=True
        )
    digest = hashlib.sha256(repr(image_ref).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


@dataclass
class APCBlock:
    """One fixed-size KV block. Holds per-layer K/V slabs once committed."""

    block_id: int
    block_hash: Optional[int] = None
    parent_hash: int = SEED_PARENT_HASH
    token_ids: Tuple[int, ...] = ()
    extra_hash: int = 0
    ref_cnt: int = 0
    keys: Optional[List[mx.array]] = None
    values: Optional[List[mx.array]] = None
    last_used: float = 0.0
    prev: Optional["APCBlock"] = None
    next: Optional["APCBlock"] = None


@dataclass
class APCStats:
    hits: int = 0
    misses: int = 0
    matched_tokens: int = 0
    served_tokens: int = 0
    evictions: int = 0
    stores: int = 0
    pool_used: int = 0

    def snapshot(self, num_blocks: int, block_size: int) -> dict:
        denom = self.matched_tokens + self.served_tokens
        hit_rate = self.matched_tokens / denom if denom > 0 else 0.0
        return {
            "block_size": block_size,
            "num_blocks": num_blocks,
            "pool_used": self.pool_used,
            "lookups_hit": self.hits,
            "lookups_miss": self.misses,
            "matched_tokens": self.matched_tokens,
            "served_tokens": self.served_tokens,
            "token_hit_rate": hit_rate,
            "evictions": self.evictions,
            "stores": self.stores,
        }


class APCManager:
    """Block pool, hash table, LRU free queue, and stats."""

    def __init__(
        self,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.pool: List[APCBlock] = [APCBlock(block_id=i) for i in range(num_blocks)]
        self._free_head: Optional[APCBlock] = None
        self._free_tail: Optional[APCBlock] = None
        for b in self.pool:
            self._free_push(b)
        self.hash_table: dict[int, APCBlock] = {}
        self.stats = APCStats()
        self.lock = threading.RLock()

    # ---------- LRU free queue (O(1)) ----------
    def _free_push(self, b: APCBlock) -> None:
        b.prev = self._free_tail
        b.next = None
        if self._free_tail is not None:
            self._free_tail.next = b
        else:
            self._free_head = b
        self._free_tail = b
        b.last_used = time.time()

    def _free_remove(self, b: APCBlock) -> None:
        if b.prev is not None:
            b.prev.next = b.next
        else:
            self._free_head = b.next
        if b.next is not None:
            b.next.prev = b.prev
        else:
            self._free_tail = b.prev
        b.prev = b.next = None

    # ---------- Block lifecycle ----------
    def _evict_lru(self) -> Optional[APCBlock]:
        b = self._free_head
        if b is None:
            return None
        self._free_remove(b)
        if b.block_hash is not None and self.hash_table.get(b.block_hash) is b:
            del self.hash_table[b.block_hash]
            self.stats.evictions += 1
        b.block_hash = None
        b.token_ids = ()
        b.parent_hash = SEED_PARENT_HASH
        b.extra_hash = 0
        b.keys = None
        b.values = None
        return b

    def _acquire_existing(self, b: APCBlock) -> APCBlock:
        if b.ref_cnt == 0:
            self._free_remove(b)
        b.ref_cnt += 1
        return b

    def _release_one(self, b: APCBlock) -> None:
        b.ref_cnt -= 1
        if b.ref_cnt <= 0:
            b.ref_cnt = 0
            self._free_push(b)

    def release(self, blocks: Iterable[APCBlock]) -> None:
        with self.lock:
            for b in blocks:
                self._release_one(b)

    # ---------- Public API ----------
    def lookup_prefix(
        self, token_ids: Sequence[int], extra_hash: int = 0
    ) -> Tuple[List[APCBlock], int]:
        """Walk the hash chain over ``token_ids``; return acquired matched
        blocks and matched_token_count. Caller must release the blocks."""
        with self.lock:
            n_full = len(token_ids) // self.block_size
            matched: List[APCBlock] = []
            parent = SEED_PARENT_HASH
            for i in range(n_full):
                chunk = tuple(
                    int(t) for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                b = self.hash_table.get(h)
                if b is None or b.token_ids != chunk:
                    break
                matched.append(self._acquire_existing(b))
                parent = h
            matched_tokens = len(matched) * self.block_size
            if matched_tokens > 0:
                self.stats.hits += 1
                self.stats.matched_tokens += matched_tokens
            else:
                self.stats.misses += 1
            return matched, matched_tokens

    def store_kv_blocks(
        self,
        token_ids: Sequence[int],
        layer_keys: List[mx.array],
        layer_values: List[mx.array],
        *,
        extra_hash: int = 0,
        skip_first_n_tokens: int = 0,
    ) -> List[APCBlock]:
        """Slice ``layer_keys`` / ``layer_values`` into block_size chunks and
        store any new full blocks beyond ``skip_first_n_tokens``.

        Returns newly acquired blocks (caller must release).
        """
        with self.lock:
            n_full = len(token_ids) // self.block_size
            skip_full = skip_first_n_tokens // self.block_size
            new_blocks: List[APCBlock] = []
            parent = SEED_PARENT_HASH
            # Recompute hash chain over already-cached prefix to get parent for first new block.
            for i in range(skip_full):
                chunk = tuple(
                    int(t) for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                parent = _hash_tokens(parent, chunk, extra_hash)

            for i in range(skip_full, n_full):
                chunk = tuple(
                    int(t) for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                existing = self.hash_table.get(h)
                if existing is not None and existing.token_ids == chunk:
                    new_blocks.append(self._acquire_existing(existing))
                    parent = h
                    continue
                b = self._evict_lru()
                if b is None:
                    logger.debug("APC pool exhausted; stopping store at block %d/%d", i, n_full)
                    break
                start = i * self.block_size
                end = start + self.block_size
                # Deep-copy each slice into its own buffer so the block tensor
                # is decoupled from the caller's cache, which mlx.clear_cache
                # may release after generation. mx.contiguous alone returns a
                # view when the source is already row-contiguous, so we add a
                # zero to force a fresh allocation through the graph.
                k_slabs = [
                    mx.contiguous(k[..., start:end, :] + mx.array(0, dtype=k.dtype))
                    for k in layer_keys
                ]
                v_slabs = [
                    mx.contiguous(v[..., start:end, :] + mx.array(0, dtype=v.dtype))
                    for v in layer_values
                ]
                mx.eval(k_slabs + v_slabs)
                b.block_hash = h
                b.parent_hash = parent
                b.token_ids = chunk
                b.extra_hash = extra_hash
                b.keys = k_slabs
                b.values = v_slabs
                b.ref_cnt = 1
                self.hash_table[h] = b
                new_blocks.append(b)
                self.stats.stores += 1
                self.stats.served_tokens += self.block_size
                parent = h
            self.stats.pool_used = sum(1 for x in self.pool if x.block_hash is not None)
            return new_blocks

    def stats_snapshot(self) -> dict:
        with self.lock:
            self.stats.pool_used = sum(1 for x in self.pool if x.block_hash is not None)
            return self.stats.snapshot(self.num_blocks, self.block_size)

    def reset_stats(self) -> None:
        with self.lock:
            self.stats = APCStats()

    def clear(self) -> None:
        with self.lock:
            for b in self.pool:
                b.block_hash = None
                b.token_ids = ()
                b.parent_hash = SEED_PARENT_HASH
                b.extra_hash = 0
                b.keys = None
                b.values = None
                b.ref_cnt = 0
                b.prev = b.next = None
            self.hash_table.clear()
            self._free_head = self._free_tail = None
            for b in self.pool:
                self._free_push(b)
            self.stats = APCStats()


def make_warm_kv_cache(
    matched_blocks: List[APCBlock],
) -> List[Any]:
    """Stitch matched blocks into per-layer ``KVCache`` instances pre-filled
    with the cached prefix's K/V state. Used by the single-stream
    ``stream_generate`` path.
    """
    from mlx_lm.models.cache import KVCache

    if not matched_blocks:
        return []
    num_layers = len(matched_blocks[0].keys)
    out: List[Any] = []
    prefix_len = sum(b.keys[0].shape[-2] for b in matched_blocks)
    for layer_idx in range(num_layers):
        ks = [b.keys[layer_idx] for b in matched_blocks]
        vs = [b.values[layer_idx] for b in matched_blocks]
        merged_k = mx.concatenate(ks, axis=2)
        merged_v = mx.concatenate(vs, axis=2)
        c = KVCache()
        c.keys = merged_k
        c.values = merged_v
        c.offset = prefix_len
        out.append(c)
    return out


def make_warm_batch_kv_cache(
    matched_blocks: List[APCBlock],
) -> List[Any]:
    """Stitch matched blocks into per-layer single-row ``BatchKVCache``
    instances pre-filled with the cached prefix's K/V state. Used by the
    batched continuous-batching path; the resulting cache list can be
    ``extend()``-ed into a running batch.
    """
    from mlx_lm.models.cache import BatchKVCache

    if not matched_blocks:
        return []
    num_layers = len(matched_blocks[0].keys)
    prefix_len = sum(b.keys[0].shape[-2] for b in matched_blocks)
    out: List[Any] = []
    for layer_idx in range(num_layers):
        ks = [b.keys[layer_idx] for b in matched_blocks]
        vs = [b.values[layer_idx] for b in matched_blocks]
        merged_k = mx.concatenate(ks, axis=2)  # [1, H, prefix_len, D]
        merged_v = mx.concatenate(vs, axis=2)
        c = BatchKVCache(left_padding=[0])
        # state setter: (keys, values, offset, left_padding) → also sets _idx
        c.state = (
            merged_k,
            merged_v,
            mx.array([prefix_len]),
            mx.array([0]),
        )
        out.append(c)
    return out


def make_warm_batch_kv_cache_multi(
    picks: List[Optional[dict]],
    num_layers: int,
) -> Tuple[List[Any], int]:
    """Build a multi-row ``BatchKVCache`` list for mixed warm / cold prefill.

    ``picks`` is per-row, with each entry being ``None`` (cold) or a dict
    with key ``matched_blocks`` (list of APCBlock) and ``prefix_len``.

    Returns ``(cache_list, max_prefix)`` where ``max_prefix`` is the cache's
    ``_idx`` after warm-init (= max prefix_len across rows).

    For row ``i``:
      * left_padding[i] = max_prefix - prefix_len[i]
      * keys[i, :, left_padding[i]:max_prefix, :] = concatenated block K
      * keys[i, :, :left_padding[i], :] = zeros (will be hidden by mask)
    """
    from mlx_lm.models.cache import BatchKVCache

    B = len(picks)
    prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
    max_prefix = max(prefix_lens) if prefix_lens else 0
    if max_prefix == 0:
        return [], 0

    # Discover dtype / head dims from the first non-empty pick.
    sample = next(p for p in picks if p)
    sample_k = sample["matched_blocks"][0].keys[0]  # [1, H, bs, D]
    n_kv_heads = sample_k.shape[1]
    head_dim = sample_k.shape[-1]
    dtype = sample_k.dtype

    out: List[Any] = []
    for layer_idx in range(num_layers):
        # Build per-row warm K/V tensors of shape [1, H, max_prefix, D]; rows
        # without a hit get zeros, rows with a shorter prefix get zero left-pad.
        row_keys: List[mx.array] = []
        row_values: List[mx.array] = []
        for i, pick in enumerate(picks):
            if pick is None:
                # Cold row: full pre-warm zone is left padding (zeros).
                row_keys.append(
                    mx.zeros((1, n_kv_heads, max_prefix, head_dim), dtype=dtype)
                )
                row_values.append(
                    mx.zeros((1, n_kv_heads, max_prefix, head_dim), dtype=dtype)
                )
                continue
            blocks = pick["matched_blocks"]
            ks = [b.keys[layer_idx] for b in blocks]
            vs = [b.values[layer_idx] for b in blocks]
            warm_k = mx.concatenate(ks, axis=2)  # [1, H, prefix_len, D]
            warm_v = mx.concatenate(vs, axis=2)
            lp = max_prefix - pick["prefix_len"]
            if lp > 0:
                pad_k = mx.zeros((1, n_kv_heads, lp, head_dim), dtype=dtype)
                pad_v = mx.zeros((1, n_kv_heads, lp, head_dim), dtype=dtype)
                warm_k = mx.concatenate([pad_k, warm_k], axis=2)
                warm_v = mx.concatenate([pad_v, warm_v], axis=2)
            row_keys.append(warm_k)
            row_values.append(warm_v)
        merged_k = mx.concatenate(row_keys, axis=0)  # [B, H, max_prefix, D]
        merged_v = mx.concatenate(row_values, axis=0)

        left_padding = [max_prefix - pl for pl in prefix_lens]
        offset = [pl for pl in prefix_lens]
        c = BatchKVCache(left_padding=[0] * B)  # placeholder; state setter overrides
        c.state = (
            merged_k,
            merged_v,
            mx.array(offset),
            mx.array(left_padding),
        )
        out.append(c)
    return out, max_prefix


def harvest_blocks_from_batch_cache(
    apc_manager: "APCManager",
    batch_caches: List[Any],
    batch_idx: int,
    full_token_ids: Sequence[int],
    *,
    extra_hash: int = 0,
    skip_first_n_tokens: int = 0,
) -> List[APCBlock]:
    """Slice one row out of a batched KV cache and store its full blocks.

    Used at the end of prompt prefill in continuous-batching mode to add
    the new prefix to APC.
    """
    layer_keys: List[mx.array] = []
    layer_values: List[mx.array] = []
    for c in batch_caches:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        idx = getattr(c, "_idx", None)
        left_padding = getattr(c, "left_padding", None)
        if keys is None or values is None or idx is None:
            return []
        # Pull this batch row, dropping any left-padding for this seq.
        if left_padding is not None:
            try:
                lp = int(left_padding[batch_idx].item())
            except Exception:
                lp = 0
        else:
            lp = 0
        # shape after slicing: [1, H, idx-lp, D]
        layer_keys.append(keys[batch_idx : batch_idx + 1, :, lp:idx, :])
        layer_values.append(values[batch_idx : batch_idx + 1, :, lp:idx, :])
    return apc_manager.store_kv_blocks(
        full_token_ids,
        layer_keys,
        layer_values,
        extra_hash=extra_hash,
        skip_first_n_tokens=skip_first_n_tokens,
    )


def model_supports_apc(language_model: Any) -> bool:
    """APC requires the default KVCache layout — opt out for custom caches."""
    return not hasattr(language_model, "make_cache")


def from_env() -> Optional[APCManager]:
    """Build an APCManager from env vars when ``APC_ENABLED=1``, else None."""
    if os.environ.get("APC_ENABLED", "0") not in ("1", "true", "True", "yes"):
        return None
    block_size = int(os.environ.get("APC_BLOCK_SIZE", DEFAULT_BLOCK_SIZE))
    num_blocks = int(os.environ.get("APC_NUM_BLOCKS", DEFAULT_NUM_BLOCKS))
    logger.info(
        "APC enabled (block_size=%d, num_blocks=%d, hash=%s)",
        block_size,
        num_blocks,
        "sha256" if _hash_use_sha256() else "fast",
    )
    return APCManager(num_blocks=num_blocks, block_size=block_size)

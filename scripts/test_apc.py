"""End-to-end test for Automatic Prefix Caching (APC).

Phase 1: pure data-structure tests that don't touch any model. They verify
hash chains, prefix lookup, eviction, and multimodal differentiation.

Phase 2 (optional): runs ``stream_generate`` against a real model and asserts
that a second call with the same prompt sees a non-zero matched-token count.
The model defaults to ``mlx-community/SmolVLM-Instruct-bf16`` (small, ~500MB)
but can be overridden with ``--model`` or the ``APC_TEST_MODEL`` env var.

Usage::

    python scripts/test_apc.py             # phase 1 only
    python scripts/test_apc.py --model-test [--model mlx-community/...]

"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from typing import List

import mlx.core as mx
import numpy as np


# Make the repo importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm.apc import (  # noqa: E402
    APCBlock,
    APCManager,
    DiskBlockStore,
    SEED_PARENT_HASH,
    _hash_tokens,
    hash_image_payload,
    make_warm_kv_cache,
)


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _info(label: str, ok: bool, detail: str = "") -> None:
    mark = _green("✓ PASS") if ok else _red("✗ FAIL")
    print(f"  {mark} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        raise AssertionError(label)


def _make_fake_kv(num_layers: int, n_kv_heads: int, seq_len: int, head_dim: int):
    """Make deterministic fake K/V tensors for ``num_layers`` layers."""
    rng = np.random.default_rng(seed=0)
    keys, values = [], []
    for _ in range(num_layers):
        k = mx.array(
            rng.standard_normal((1, n_kv_heads, seq_len, head_dim)).astype(np.float32)
        )
        v = mx.array(
            rng.standard_normal((1, n_kv_heads, seq_len, head_dim)).astype(np.float32)
        )
        keys.append(k)
        values.append(v)
    mx.eval(keys + values)
    return keys, values


# --------------------------- Phase 1 unit tests ---------------------------


def test_hash_determinism() -> None:
    print("\n[1/8] Hash determinism")
    a = _hash_tokens(0, tuple(range(16)), 0)
    b = _hash_tokens(0, tuple(range(16)), 0)
    _info("hash deterministic", a == b)
    _info(
        "extra_hash differentiates",
        _hash_tokens(0, tuple(range(16)), 0) != _hash_tokens(0, tuple(range(16)), 42),
    )
    _info(
        "parent differentiates",
        _hash_tokens(7, tuple(range(16)), 0) != _hash_tokens(8, tuple(range(16)), 0),
    )


def test_image_hash() -> None:
    print("\n[2/8] Multimodal hash")
    arr1 = mx.array(np.zeros((1, 3, 32, 32), dtype=np.float32))
    arr2 = mx.array(np.ones((1, 3, 32, 32), dtype=np.float32))
    h1 = hash_image_payload(pixel_values=arr1, image_ref=None)
    h2 = hash_image_payload(pixel_values=arr2, image_ref=None)
    _info("different pixels → different hash", h1 != h2)
    _info("zero hash for empty input", hash_image_payload(None, None) == 0)
    _info(
        "same path → same hash",
        hash_image_payload(image_ref="cat.jpg") == hash_image_payload(image_ref="cat.jpg"),
    )


def test_lookup_and_store() -> None:
    print("\n[3/8] Block lookup + store")
    bs = 16
    mgr = APCManager(num_blocks=64, block_size=bs)

    n_layers, n_heads, head_dim = 4, 2, 32
    seq_len = 3 * bs  # exactly 3 full blocks
    tokens = list(range(seq_len))
    layer_keys, layer_values = _make_fake_kv(n_layers, n_heads, seq_len, head_dim)

    # First lookup → miss
    m, n = mgr.lookup_prefix(tokens)
    _info("first lookup misses", n == 0 and m == [])

    # Store
    new_blocks = mgr.store_kv_blocks(tokens, layer_keys, layer_values, skip_first_n_tokens=0)
    _info("3 blocks stored", len(new_blocks) == 3)
    mgr.release(new_blocks)

    # Second lookup → 3 blocks
    m2, n2 = mgr.lookup_prefix(tokens)
    _info("second lookup hits 3 blocks", len(m2) == 3 and n2 == 3 * bs)

    # Make warm cache and verify shapes
    warm = make_warm_kv_cache(m2)
    _info("warm cache has correct layer count", len(warm) == n_layers)
    _info(
        "warm cache offset = 3*bs",
        all(c.offset == 3 * bs for c in warm),
    )
    _info(
        "warm cache shape correct",
        all(c.keys.shape == (1, n_heads, 3 * bs, head_dim) for c in warm),
    )

    # Partial-prefix mismatch: alter the 3rd block's tokens → should match 2 blocks
    altered = list(tokens)
    altered[2 * bs] = 9999  # token in 3rd block
    m3, n3 = mgr.lookup_prefix(altered)
    _info("altered token caps prefix at 2 blocks", n3 == 2 * bs)
    mgr.release(m2)
    mgr.release(m3)


def test_eviction_lru() -> None:
    print("\n[4/8] LRU eviction")
    bs = 16
    mgr = APCManager(num_blocks=4, block_size=bs)  # tiny pool
    n_layers, n_heads, head_dim = 2, 1, 8

    # Fill 4 blocks worth of unique tokens (4 chains)
    chains = []
    for offset in (0, 1000, 2000, 3000):
        toks = list(range(offset, offset + bs))
        keys, vals = _make_fake_kv(n_layers, n_heads, bs, head_dim)
        nb = mgr.store_kv_blocks(toks, keys, vals)
        mgr.release(nb)
        chains.append(toks)

    snap = mgr.stats_snapshot()
    _info("pool fully used", snap["pool_used"] == 4)

    # Touch chain 0 to make it MRU (lookup → released → moved to tail)
    m0, _ = mgr.lookup_prefix(chains[0])
    mgr.release(m0)

    # Add a 5th unique chain → should evict LRU (chain 1 is now LRU because
    # 0,2,3 were just released — ordering: 1 is at head; let's verify chain 1
    # is the one evicted).
    new_toks = list(range(9000, 9000 + bs))
    keys, vals = _make_fake_kv(n_layers, n_heads, bs, head_dim)
    nb = mgr.store_kv_blocks(new_toks, keys, vals)
    mgr.release(nb)

    snap = mgr.stats_snapshot()
    _info("at least one eviction occurred", snap["evictions"] >= 1)
    _info("pool still fully used", snap["pool_used"] == 4)

    # Chain 0 should still be present (it was MRU)
    m, n = mgr.lookup_prefix(chains[0])
    _info("MRU chain survives eviction", n == bs)
    mgr.release(m)


def test_multimodal_collision_avoidance() -> None:
    print("\n[5/8] Image-hash isolation")
    bs = 16
    mgr = APCManager(num_blocks=16, block_size=bs)
    toks = list(range(bs * 2))
    n_layers, n_heads, head_dim = 2, 1, 8
    keys, vals = _make_fake_kv(n_layers, n_heads, bs * 2, head_dim)

    img_a = hash_image_payload(image_ref="cat.jpg")
    img_b = hash_image_payload(image_ref="dog.jpg")

    # Store under image A
    nb_a = mgr.store_kv_blocks(toks, keys, vals, extra_hash=img_a)
    _info("stored under image A", len(nb_a) == 2)
    mgr.release(nb_a)

    # Lookup under image B → must miss
    m_b, n_b = mgr.lookup_prefix(toks, extra_hash=img_b)
    _info("different image → no hit", n_b == 0)

    # Lookup under image A → must hit
    m_a, n_a = mgr.lookup_prefix(toks, extra_hash=img_a)
    _info("same image → full hit", n_a == 2 * bs)
    mgr.release(m_a)


def test_partial_block_not_stored() -> None:
    print("\n[6/8] Partial trailing block ignored")
    bs = 16
    mgr = APCManager(num_blocks=16, block_size=bs)
    toks = list(range(bs + 5))  # 1 full + 5 leftover
    n_layers, n_heads, head_dim = 2, 1, 8
    keys, vals = _make_fake_kv(n_layers, n_heads, bs + 5, head_dim)
    nb = mgr.store_kv_blocks(toks, keys, vals)
    _info("only the full block is stored", len(nb) == 1)
    mgr.release(nb)


def test_skip_first_n() -> None:
    print("\n[7/8] skip_first_n_tokens (no double-store)")
    bs = 16
    mgr = APCManager(num_blocks=16, block_size=bs)
    n_layers, n_heads, head_dim = 2, 1, 8
    toks = list(range(2 * bs))
    keys, vals = _make_fake_kv(n_layers, n_heads, 2 * bs, head_dim)

    # First store: both blocks
    nb1 = mgr.store_kv_blocks(toks, keys, vals)
    _info("initial store has 2 blocks", len(nb1) == 2)
    mgr.release(nb1)

    stores_before = mgr.stats_snapshot()["stores"]

    # Second call simulating warm-start: skip first block (already cached).
    # Should NOT add a new block (the 2nd block is already cached, hash hits).
    nb2 = mgr.store_kv_blocks(toks, keys, vals, skip_first_n_tokens=bs)
    stores_after = mgr.stats_snapshot()["stores"]
    _info(
        "no new evictions/stores when block already cached",
        stores_after == stores_before and len(nb2) == 1,
    )
    mgr.release(nb2)


def test_disk_layer_major_warm_prefix() -> None:
    print("\n[8/8] Disk layer-major warm prefix")
    bs = 16
    root = tempfile.mkdtemp(prefix="apc-disk-test-")
    try:
        n_layers, n_heads, head_dim = 2, 1, 8
        seq_len = 3 * bs
        toks = list(range(seq_len))
        keys, vals = _make_fake_kv(n_layers, n_heads, seq_len, head_dim)

        disk = DiskBlockStore(root, namespace="unit")
        mgr = APCManager(num_blocks=1, block_size=bs, disk=disk)
        nb = mgr.store_kv_blocks(toks, keys, vals)
        _info("memory pool stores bounded subset", len(nb) == 1)
        mgr.release(nb)
        disk.close()

        disk = DiskBlockStore(root, namespace="unit")
        mgr = APCManager(num_blocks=16, block_size=bs, disk=disk)
        warm, matched = mgr.lookup_prefix_disk_cache(toks)
        _info(
            "disk restore returns full prefix",
            warm is not None and matched == seq_len,
        )
        _info("warm cache has correct layer count", len(warm) == n_layers)
        _info(
            "warm cache offset is exact prefix",
            all(c.offset == seq_len for c in warm),
        )
        _info(
            "warm cache preserves spare capacity",
            all(c.keys.shape[2] > c.offset for c in warm),
        )
        _info(
            "disk restore does not populate APCBlock pool",
            mgr.stats_snapshot()["pool_used"] == 0,
        )
        disk.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------- Phase 2 model test ---------------------------


def test_with_model(model_path: str, max_tokens: int = 16) -> None:
    print(f"\n[model] Running stream_generate with APC against {model_path}")
    from mlx_vlm import load
    from mlx_vlm.apc import APCManager
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    print("  loading model …")
    t0 = time.time()
    model, processor = load(model_path)
    print(f"  loaded in {time.time()-t0:.1f}s")
    config = model.config

    if hasattr(model.language_model, "make_cache"):
        print(_red("  model uses custom KVCache layout — APC won't activate"))
        return

    apc_mgr = APCManager(num_blocks=4096, block_size=16)
    prompt_text = (
        "List exactly five facts about transformer language models, "
        "one per line, beginning each with a dash."
    )
    formatted = apply_chat_template(processor, config, prompt=prompt_text, num_images=0)

    def _run(label: str):
        t0 = time.time()
        last = None
        for chunk in stream_generate(
            model=model,
            processor=processor,
            prompt=formatted,
            max_tokens=max_tokens,
            temperature=0.0,
            apc_manager=apc_mgr,
        ):
            last = chunk
        elapsed = time.time() - t0
        snap = apc_mgr.stats_snapshot()
        print(
            f"  [{label}] elapsed={elapsed:.2f}s prompt_tps={last.prompt_tps:.0f} "
            f"matched_tokens={snap['matched_tokens']} stores={snap['stores']}"
        )
        return last, snap

    apc_mgr.reset_stats()
    _info("APC pool starts empty", apc_mgr.stats_snapshot()["pool_used"] == 0)

    out1, s1 = _run("first call")
    _info("first call stored at least 1 block", s1["stores"] >= 1)

    out2, s2 = _run("second call (same prompt)")
    _info(
        "second call matches the cached prefix",
        s2["matched_tokens"] >= 16,
        f"matched_tokens={s2['matched_tokens']}",
    )

    # Run with a *different* prompt → no match
    different = apply_chat_template(
        processor,
        config,
        prompt="What is the capital of France? Reply in one word.",
        num_images=0,
    )
    apc_mgr.reset_stats()
    list(
        stream_generate(
            model=model,
            processor=processor,
            prompt=different,
            max_tokens=max_tokens,
            temperature=0.0,
            apc_manager=apc_mgr,
        )
    )
    s3 = apc_mgr.stats_snapshot()
    _info("different prompt → 0 matched tokens", s3["matched_tokens"] == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-test",
        action="store_true",
        help="Also run the end-to-end test with a real model.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("APC_TEST_MODEL", "mlx-community/SmolVLM-Instruct-bf16"),
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    print("=" * 60)
    print("APC unit tests")
    print("=" * 60)

    test_hash_determinism()
    test_image_hash()
    test_lookup_and_store()
    test_eviction_lru()
    test_multimodal_collision_avoidance()
    test_partial_block_not_stored()
    test_skip_first_n()
    test_disk_layer_major_warm_prefix()

    print()
    print(_green("All unit tests passed."))

    if args.model_test:
        print()
        print("=" * 60)
        print("Model integration test")
        print("=" * 60)
        test_with_model(args.model, max_tokens=args.max_tokens)
        print(_green("Model test passed."))


if __name__ == "__main__":
    main()

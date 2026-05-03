"""Profile APC segment-shard disk writes, restores, and eviction.

This is a synthetic no-model harness for testing very long APC disk sessions
without paying full model prefill cost. It creates Qwen3-VL-shaped layer-major
KV tensors, stores them through ``APCManager.store_kv_blocks``, waits for the
disk writer, then checks how much of each session remains after disk-cap
eviction.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mlx.core as mx
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm.apc import (  # noqa: E402
    SEED_PARENT_HASH,
    APCManager,
    DiskBlockStore,
    _hash_tokens,
)

N_LAYERS = 36
N_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = mx.bfloat16


@dataclass
class SessionInfo:
    idx: int
    tokens: List[int]
    hashes: List[int]


def gb(n: float) -> float:
    return n / (1 << 30)


def mlx_mem() -> str:
    return (
        f"active={gb(mx.get_active_memory()):6.2f}GB "
        f"cache={gb(mx.get_cache_memory()):6.2f}GB "
        f"peak={gb(mx.get_peak_memory()):6.2f}GB "
        f"rss={gb(psutil.Process().memory_info().rss):6.2f}GB "
        f"avail={gb(psutil.virtual_memory().available):6.1f}GB"
    )


def make_tokens(session_idx: int, n_tokens: int) -> List[int]:
    base = (session_idx + 1) * 10_000_000
    return [base + i for i in range(n_tokens)]


def hash_blocks(tokens: List[int]) -> List[int]:
    parent = SEED_PARENT_HASH
    out: List[int] = []
    for i in range(len(tokens) // BLOCK_SIZE):
        chunk = tuple(tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE])
        parent = _hash_tokens(parent, chunk, 0)
        out.append(parent)
    return out


def make_layers(n_tokens: int):
    shape = (1, N_KV_HEADS, n_tokens, HEAD_DIM)
    keys = [mx.zeros(shape, dtype=DTYPE) for _ in range(N_LAYERS)]
    values = [mx.zeros(shape, dtype=DTYPE) for _ in range(N_LAYERS)]
    mx.eval(keys + values)
    return keys, values


def prefix_present_blocks(disk: DiskBlockStore, hashes: List[int]) -> int:
    n = 0
    for h in hashes:
        if not disk.has(h):
            break
        n += 1
    return n


def count_present_blocks(disk: DiskBlockStore, hashes: List[int]) -> int:
    return sum(1 for h in hashes if disk.has(h))


def shard_count(disk: DiskBlockStore) -> int:
    return sum(
        1
        for p in disk.dir.glob(f"*{disk.SUFFIX}")
        if p.is_file() and disk._is_canonical_shard(p)
    )


def write_session(
    disk: DiskBlockStore,
    session_idx: int,
    n_tokens: int,
) -> SessionInfo:
    tokens = make_tokens(session_idx, n_tokens)
    hashes = hash_blocks(tokens)
    print(
        f"\n[session {session_idx}] materialize {n_tokens:,} tokens "
        f"({len(hashes):,} blocks)",
        flush=True,
    )
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    keys, values = make_layers(n_tokens)
    print(
        f"  source KV materialized in {time.perf_counter() - t0:.2f}s  {mlx_mem()}",
        flush=True,
    )

    mgr = APCManager(num_blocks=1, block_size=BLOCK_SIZE, disk=disk)
    t0 = time.perf_counter()
    nb = mgr.store_kv_blocks(tokens, keys, values)
    schedule_s = time.perf_counter() - t0
    mgr.release(nb)

    t0 = time.perf_counter()
    disk._q.join()
    flush_s = time.perf_counter() - t0
    snap = mgr.stats_snapshot()
    print(
        "  write "
        f"scheduled={schedule_s:.2f}s flushed={flush_s:.2f}s "
        f"files={snap.get('disk_files')} blocks_indexed={snap.get('disk_blocks_indexed')} "
        f"evictions={snap.get('disk_evictions')} bytes={gb(snap.get('disk_bytes', 0)):.2f}GB",
        flush=True,
    )

    del keys, values, nb, mgr
    gc.collect()
    mx.clear_cache()
    print(f"  after source drop      {mlx_mem()}", flush=True)
    return SessionInfo(session_idx, tokens, hashes)


def restore_session(disk: DiskBlockStore, info: SessionInfo) -> None:
    present = count_present_blocks(disk, info.hashes)
    prefix = prefix_present_blocks(disk, info.hashes)
    print(
        f"\n[restore session {info.idx}] present={present:,}/{len(info.hashes):,} "
        f"prefix={prefix:,} blocks ({prefix * BLOCK_SIZE:,} tokens)",
        flush=True,
    )
    if prefix == 0:
        return
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    loaded = disk.load_layer_major_prefix(info.hashes[:prefix])
    restore_s = time.perf_counter() - t0
    if loaded is None:
        print(f"  restore failed after {restore_s:.2f}s", flush=True)
        return
    keys, values, metadata = loaded
    capacity = int(keys[0].shape[2]) if keys else 0
    print(
        f"  restored={len(metadata):,} blocks wall={restore_s:.2f}s "
        f"capacity_tokens={capacity:,} {mlx_mem()}",
        flush=True,
    )
    del keys, values, metadata, loaded
    gc.collect()
    mx.clear_cache()
    print(f"  after restore drop     {mlx_mem()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--disk-cap-gb", type=float, default=24.0)
    ap.add_argument("--shard-max-blocks", type=int, default=256)
    ap.add_argument("--disk-path", default=None)
    ap.add_argument("--keep-disk", action="store_true")
    args = ap.parse_args()

    os.environ["APC_DISK_SHARD_MAX_BLOCKS"] = str(max(1, args.shard_max_blocks))
    max_bytes = int(args.disk_cap_gb * (1 << 30)) if args.disk_cap_gb > 0 else None
    root = (
        Path(args.disk_path)
        if args.disk_path
        else Path(tempfile.mkdtemp(prefix="apc-segment-profile-"))
    )
    root.mkdir(parents=True, exist_ok=True)

    print("APC segment eviction profile", flush=True)
    print(
        f"  sessions={args.sessions} tokens/session={args.tokens:,} "
        f"blocks/session={args.tokens // BLOCK_SIZE:,}",
        flush=True,
    )
    print(
        f"  dims={N_LAYERS} layers x {N_KV_HEADS} kv-heads x {HEAD_DIM} "
        f"dtype={DTYPE} block_size={BLOCK_SIZE}",
        flush=True,
    )
    print(
        f"  shard_max_blocks={args.shard_max_blocks} cap={args.disk_cap_gb:g}GB "
        f"root={root}",
        flush=True,
    )
    print(f"  start memory           {mlx_mem()}", flush=True)

    infos: List[SessionInfo] = []
    disk = DiskBlockStore(root, namespace="segment-profile", max_bytes=max_bytes)
    try:
        for session_idx in range(args.sessions):
            infos.append(write_session(disk, session_idx, args.tokens))
            print(
                f"  post-session {session_idx}: files={shard_count(disk)} "
                f"indexed={disk.num_blocks_indexed:,} bytes={gb(disk.disk_bytes):.2f}GB "
                f"evictions={disk.evictions}",
                flush=True,
            )

        print("\n[presence after writes]", flush=True)
        for info in infos:
            present = count_present_blocks(disk, info.hashes)
            prefix = prefix_present_blocks(disk, info.hashes)
            print(
                f"  session {info.idx}: present={present:,}/{len(info.hashes):,} "
                f"prefix={prefix:,} blocks ({prefix * BLOCK_SIZE:,} tokens)",
                flush=True,
            )

        for info in infos:
            restore_session(disk, info)

        print(
            f"\n[final] files={shard_count(disk)} indexed={disk.num_blocks_indexed:,} "
            f"bytes={gb(disk.disk_bytes):.2f}GB evictions={disk.evictions}",
            flush=True,
        )
    finally:
        disk.close()
        if args.disk_path is None and not args.keep_disk:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"Disk dir: {root}", flush=True)


if __name__ == "__main__":
    main()

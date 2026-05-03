"""Focused profiler for the APC disk read path. No model loaded; synthetic
blocks with Qwen3-VL-4B-Instruct dimensions (36 layers × 8 KV-heads ×
head_dim 128 × bf16). Goal: localise the cost / memory-pressure source so
we can stop guessing.

Stages profiled in isolation:
  1. ``DiskBlockStore.load`` — direct file read by default, or mmap fallback
  2. Deep-copy (`+ mx.array(0)` + `mx.contiguous`) — does it actually sever
     the result from the mmap region?
  3. ``mx.eval`` at various coalesce widths (1, 4, 8, 16 blocks)
  4. Re-access after dropping the mmap cache — does the materialised data
     survive, or does it re-fault from disk?

Each stage prints wall time + RSS + system free RAM. Tiny block count
(50 blocks ≈ 115 MB) so this is safe to run alongside other workloads.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_vlm.apc import APCManager, DiskBlockStore, _hash_tokens

# Qwen3-VL-4B-Instruct attention dims
N_LAYERS = 36
N_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = mx.bfloat16

PER_TENSOR_BYTES = 1 * N_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2  # bf16
PER_BLOCK_BYTES = N_LAYERS * 2 * PER_TENSOR_BYTES  # K + V


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def avail_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)


def mlx_mem_mb() -> tuple[float, float, float]:
    return (
        mx.get_active_memory() / 1024 / 1024,
        mx.get_cache_memory() / 1024 / 1024,
        mx.get_peak_memory() / 1024 / 1024,
    )


def mem_line() -> str:
    active, cache, peak = mlx_mem_mb()
    return (
        f"rss={rss_mb():7.1f}MB  active={active:7.1f}MB  "
        f"cache={cache:7.1f}MB  peak={peak:7.1f}MB  avail={avail_gb():5.1f}GB"
    )


def fmt_ms(ns_list):
    if not ns_list:
        return "n=0"
    arr = np.array(ns_list) / 1e6
    return (
        f"n={len(arr):3d}  "
        f"med={np.median(arr):7.2f}ms  "
        f"max={np.max(arr):7.2f}ms  "
        f"sum={np.sum(arr):8.1f}ms"
    )


def setup_disk(n_blocks: int, root: Path) -> list:
    """Write n_blocks via store_kv_blocks; tear down; return token_ids."""
    print(f"\n[setup] writing {n_blocks} blocks to {root} …")
    disk = DiskBlockStore(root, namespace="profile")
    mgr = APCManager(num_blocks=n_blocks + 32, block_size=BLOCK_SIZE, disk=disk)

    rng = np.random.default_rng(42)
    token_ids = rng.integers(0, 100_000, n_blocks * BLOCK_SIZE).tolist()

    # Build the full prefix as one slab per layer (mirrors how a real
    # prefill would produce K/V).
    n_tok = n_blocks * BLOCK_SIZE
    layer_keys = []
    layer_values = []
    for _ in range(N_LAYERS):
        k = mx.array(
            rng.standard_normal((1, N_KV_HEADS, n_tok, HEAD_DIM)).astype(np.float32)
        ).astype(DTYPE)
        v = mx.array(
            rng.standard_normal((1, N_KV_HEADS, n_tok, HEAD_DIM)).astype(np.float32)
        ).astype(DTYPE)
        layer_keys.append(k)
        layer_values.append(v)
    mx.eval(layer_keys + layer_values)

    t0 = time.perf_counter()
    nb = mgr.store_kv_blocks(token_ids, layer_keys, layer_values)
    print(f"  store_kv_blocks: {len(nb)} blocks in {time.perf_counter()-t0:.2f}s")
    mgr.release(nb)
    del layer_keys, layer_values
    gc.collect()

    disk.close()
    time.sleep(1.0)
    return token_ids


def profile_load(
    token_ids: list,
    root: Path,
    *,
    chunk_blocks: int,
    deep_copy: bool,
    label: str,
):
    print(f"\n{'='*72}")
    print(f"{label}  (chunk={chunk_blocks}  deep_copy={deep_copy})")
    print(f"{'='*72}")
    mx.reset_peak_memory()
    print(f"[init]                    {mem_line()}")

    # Fresh store — forces _rebuild_index. Tear-down at the end frees the
    # mmap cache, exposing whether holding only APCBlock refs is enough.
    disk = DiskBlockStore(root, namespace="profile")
    print(f"[after open]              {mem_line()}  indexed={disk.num_blocks_indexed}")

    n_full = len(token_ids) // BLOCK_SIZE
    parent = 0  # SEED_PARENT_HASH

    times_load = []
    times_copy = []
    times_eval = []
    blocks_kept: list = []  # hold APCBlock-equivalent refs to test re-access

    pending = []
    pending_blocks = 0

    rss_before_loop = rss_mb()
    t_total = time.perf_counter_ns()

    for i in range(n_full):
        chunk = tuple(int(t) for t in token_ids[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE])
        h = _hash_tokens(parent, chunk, 0)
        if not disk.has(h):
            break

        t0 = time.perf_counter_ns()
        loaded = disk.load(h)
        t1 = time.perf_counter_ns()
        times_load.append(t1 - t0)
        if loaded is None:
            break
        keys, values, _md = loaded

        if deep_copy:
            t0 = time.perf_counter_ns()
            keys = [mx.contiguous(k + mx.array(0, dtype=k.dtype)) for k in keys]
            values = [mx.contiguous(v + mx.array(0, dtype=v.dtype)) for v in values]
            t1 = time.perf_counter_ns()
            times_copy.append(t1 - t0)

        pending.extend(keys)
        pending.extend(values)
        pending_blocks += 1
        blocks_kept.append((keys, values))

        if pending_blocks >= chunk_blocks:
            t0 = time.perf_counter_ns()
            mx.eval(pending)
            t1 = time.perf_counter_ns()
            times_eval.append(t1 - t0)
            pending.clear()
            pending_blocks = 0

        parent = h

    if pending:
        t0 = time.perf_counter_ns()
        mx.eval(pending)
        t1 = time.perf_counter_ns()
        times_eval.append(t1 - t0)

    wall_s = (time.perf_counter_ns() - t_total) / 1e9
    rss_after_loop = rss_mb()
    active_after, cache_after, _peak_after = mlx_mem_mb()
    print(
        f"[after load+eval]         {mem_line()}  "
        f"Δrss={rss_after_loop-rss_before_loop:+7.1f}MB"
    )
    print(f"                          active+cache={active_after+cache_after:7.1f}MB")
    print(f"  wall                    {wall_s:.2f}s for {len(blocks_kept)} blocks")
    print(f"  DiskBlockStore.load     {fmt_ms(times_load)}")
    if times_copy:
        print(f"  deep-copy (+0)          {fmt_ms(times_copy)}")
    print(f"  mx.eval                 {fmt_ms(times_eval)}")

    # Re-access pattern: do the materialised K/V tensors stay device-
    # resident, or does access fault back through mmap? We measure first
    # before closing the disk store, then after closing it (which clears
    # the mmap cache and allows the source mmaps to drop if no APCBlock-
    # equivalent ref keeps them alive).
    def re_access(label, n=3):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            r = (blocks_kept[0][0][0] * 2).sum()
            mx.eval(r)
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        print(f"  re-access {label:<22} ms={[f'{t:6.2f}' for t in ts]}")

    re_access("(disk store still open)")

    print("\n  closing disk store …")
    disk.close()
    del disk
    gc.collect()
    print(f"[after disk.close]        {mem_line()}")

    re_access("(disk store closed)")

    # Touch every cached K/V tensor to confirm none page-fault unexpectedly.
    t0 = time.perf_counter_ns()
    sums = []
    for keys, values in blocks_kept:
        for k in keys:
            sums.append(k.sum())
        for v in values:
            sums.append(v.sum())
    mx.eval(sums)
    t1 = time.perf_counter_ns()
    print(
        f"  touch-all sum             {(t1-t0)/1e6:.2f}ms  "
        f"({len(sums)} tensors, {len(blocks_kept)*PER_BLOCK_BYTES/1024/1024:.1f}MB)"
    )

    del blocks_kept
    gc.collect()
    print(f"[after drop blocks]       {mem_line()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=50)
    args = ap.parse_args()

    print(f"profile target")
    print(
        f"  layers={N_LAYERS}  heads={N_KV_HEADS}  head_dim={HEAD_DIM}  bs={BLOCK_SIZE}  dtype={DTYPE}"
    )
    print(f"  per-tensor : {PER_TENSOR_BYTES/1024:.1f}KB")
    print(f"  per-block  : {PER_BLOCK_BYTES/1024:.1f}KB ({N_LAYERS*2} tensors)")
    print(f"  n_blocks   : {args.n_blocks}")
    print(f"  total      : {args.n_blocks*PER_BLOCK_BYTES/1024/1024:.1f}MB")
    print(f"  read_mode  : {os.environ.get('APC_DISK_READ_MODE', 'direct')}")
    print(f"  rss start  : {rss_mb():.1f}MB")
    print(f"  avail start: {avail_gb():.1f}GB")

    root = Path(tempfile.mkdtemp(prefix="apc-profile-"))
    try:
        token_ids = setup_disk(args.n_blocks, root)
        time.sleep(1.0)
        gc.collect()

        for label, chunk, copy in [
            ("Variant A — 1 block / no deep-copy", 1, False),
            ("Variant B — 1 block / + deep-copy", 1, True),
            ("Variant C — 4 blocks / + deep-copy", 4, True),
            ("Variant D — 8 blocks / + deep-copy", 8, True),
            ("Variant E — 16 blocks / + deep-copy", 16, True),
        ]:
            profile_load(
                token_ids, root, chunk_blocks=chunk, deep_copy=copy, label=label
            )
            time.sleep(0.5)
            gc.collect()
            time.sleep(0.5)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        print(f"\n[done] rss={rss_mb():.1f}MB  avail={avail_gb():.1f}GB")


if __name__ == "__main__":
    main()

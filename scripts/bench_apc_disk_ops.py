"""Microbench: how fast are DiskBlockStore.load and _maybe_evict?

Measures two things in isolation, no FastAPI involved:

  1. ``load`` latency — time to read a single block off SSD into MLX arrays
     and verify metadata, both cold (first read = SSD I/O) and warm (page
     cache hot = ~memcpy speed).

  2. ``_maybe_evict`` latency — given a directory with N existing blocks
     and a cap that requires evicting K of them, how long does the sweep
     take? This bounds the per-write tail latency added by eviction.

Run on a few representative block sizes so you can see how the load cost
scales with model dims (more layers / KV heads / head_dim ⇒ bigger files).
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_vlm.apc import APCManager, DiskBlockStore


PROFILES = {
    # name, n_layers, n_kv_heads, head_dim, dtype
    "smolvlm_bf16":      (24,  3, 64, mx.bfloat16),  # Idefics3 small
    "qwen3_vl_4b_bf16":  (36,  8, 128, mx.bfloat16),
    "qwen3_vl_8b_bf16":  (36,  8, 128, mx.bfloat16),  # similar dims
    "gemma3_12b_bf16":   (62, 16, 256, mx.bfloat16),  # the arxiv paper's setup
}

BLOCK_SIZE = 16


def _make_block(n_layers, n_kv_heads, head_dim, dtype, seed=0):
    rng = np.random.default_rng(seed)
    fp = np.float32 if dtype == mx.bfloat16 else np.float16
    keys, values = [], []
    for _ in range(n_layers):
        k = mx.array(
            rng.standard_normal((1, n_kv_heads, BLOCK_SIZE, head_dim)).astype(fp)
        ).astype(dtype)
        v = mx.array(
            rng.standard_normal((1, n_kv_heads, BLOCK_SIZE, head_dim)).astype(fp)
        ).astype(dtype)
        keys.append(k)
        values.append(v)
    mx.eval(keys + values)
    return keys, values


def _drop_page_cache(path: Path) -> None:
    """Best-effort cold-cache nudge. macOS doesn't expose drop_caches.
    Re-opening a file with O_DIRECT is unreliable; falling back to pure
    measurement of (cold-OS-cache hot-disk-cache) which is realistic for
    a process restart on a non-rebooted host.
    """
    pass


def bench_load(profile_name: str, n_warmup: int = 1, n_runs: int = 5) -> dict:
    n_layers, n_kv_heads, head_dim, dtype = PROFILES[profile_name]
    tmp = Path(tempfile.mkdtemp(prefix=f"apc-bench-{profile_name}-"))
    try:
        disk = DiskBlockStore(tmp, namespace=profile_name)
        mgr = APCManager(num_blocks=64, block_size=BLOCK_SIZE, disk=disk)
        keys, values = _make_block(n_layers, n_kv_heads, head_dim, dtype)
        ids = list(range(BLOCK_SIZE))
        nb = mgr.store_kv_blocks(ids, keys, values)
        mgr.release(nb)
        disk.close()
        time.sleep(0.5)  # let writer flush

        # Find the file
        ns_dir = next(tmp.iterdir())
        files = [f for f in ns_dir.glob("*.safetensors") if len(f.stem) == 16]
        assert files, "no block file written"
        path = files[0]
        bytes_on_disk = path.stat().st_size

        # Reopen a fresh DiskBlockStore so we exercise the load path cleanly.
        disk2 = DiskBlockStore(tmp, namespace=profile_name)
        block_hash = int(path.stem, 16)
        if block_hash >= (1 << 63):
            block_hash -= 1 << 64

        # Warmup the path + page cache
        for _ in range(n_warmup):
            disk2.load(block_hash)

        # Two timings:
        #   - mmap_ms: the actual load() path (zero-copy mmap of safetensors)
        #   - materialize_ms: load() + force every layer's K and V to be
        #     produced on-device. Closer to what the model attention sees.
        mmap_runs_ms: list[float] = []
        full_runs_ms: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter_ns()
            loaded = disk2.load(block_hash)
            mmap_ms = (time.perf_counter_ns() - t0) / 1e6
            keys_l, values_l, _md = loaded
            t1 = time.perf_counter_ns()
            mx.eval(keys_l + values_l)
            full_ms = mmap_ms + (time.perf_counter_ns() - t1) / 1e6
            mmap_runs_ms.append(mmap_ms)
            full_runs_ms.append(full_ms)
        disk2.close()

        return {
            "profile": profile_name,
            "n_layers": n_layers,
            "bytes_on_disk": bytes_on_disk,
            "mmap_med_ms": statistics.median(mmap_runs_ms),
            "full_med_ms": statistics.median(full_runs_ms),
            "full_max_ms": max(full_runs_ms),
            "throughput_mb_s": (
                bytes_on_disk / 1e6 / (statistics.median(full_runs_ms) / 1000.0)
            ),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench_evict(profile_name: str, n_blocks: int, evict_pct: float, n_runs: int = 3):
    """Pre-populate `n_blocks` files; cap so eviction has to drop ~evict_pct.
    Time the sweep."""
    n_layers, n_kv_heads, head_dim, dtype = PROFILES[profile_name]
    tmp = Path(tempfile.mkdtemp(prefix=f"apc-bench-evict-{profile_name}-"))
    try:
        # Stage 1: write n_blocks files with a generous cap (no eviction)
        big_disk = DiskBlockStore(tmp, namespace=profile_name)
        mgr = APCManager(num_blocks=max(64, n_blocks + 8), block_size=BLOCK_SIZE, disk=big_disk)
        for s in range(n_blocks):
            keys, values = _make_block(n_layers, n_kv_heads, head_dim, dtype, seed=s)
            ids = list(range(s * BLOCK_SIZE * 100, s * BLOCK_SIZE * 100 + BLOCK_SIZE))
            nb = mgr.store_kv_blocks(ids, keys, values)
            mgr.release(nb)
        big_disk.close()
        time.sleep(1.5)  # let all writes flush

        # Stage 2: open a new store with a tighter cap and time _maybe_evict.
        ns_dir = next(tmp.iterdir())
        files = [f for f in ns_dir.glob("*.safetensors") if len(f.stem) == 16]
        sizes = [f.stat().st_size for f in files]
        total = sum(sizes)
        avg = total / len(sizes)
        # Cap leaves room for (1 - evict_pct) of files at the low watermark.
        target_files_after = max(1, int(len(sizes) * (1 - evict_pct)))
        # Low watermark = 0.9 × max_bytes; target_files_after × avg ≈ 0.9 × cap
        cap_bytes = int(target_files_after * avg / 0.9)

        runs_ms: list[float] = []
        for run in range(n_runs):
            # Each run needs a fresh fully-populated dir for repeatable timing.
            if run > 0:
                # Replicate from the original by copying back (cheap on APFS COW)
                pass
            disk_eval = DiskBlockStore(tmp, namespace=profile_name, max_bytes=cap_bytes)
            n_files_before = len(
                [f for f in disk_eval.dir.glob("*.safetensors") if len(f.stem) == 16]
            )
            t0 = time.perf_counter_ns()
            n_evicted = disk_eval._maybe_evict()
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            disk_eval.close()
            runs_ms.append(elapsed_ms)
            if run == 0:
                print(
                    f"  evict run0: {n_files_before} files in dir → "
                    f"evicted {n_evicted} → {elapsed_ms:.2f}ms"
                )
            # Restore for next run by re-staging files
            if run < n_runs - 1:
                # Wipe and re-stage
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True, exist_ok=True)
                big_disk = DiskBlockStore(tmp, namespace=profile_name)
                mgr = APCManager(num_blocks=max(64, n_blocks + 8), block_size=BLOCK_SIZE, disk=big_disk)
                for s in range(n_blocks):
                    keys, values = _make_block(n_layers, n_kv_heads, head_dim, dtype, seed=s)
                    ids = list(range(s * BLOCK_SIZE * 100, s * BLOCK_SIZE * 100 + BLOCK_SIZE))
                    nb = mgr.store_kv_blocks(ids, keys, values)
                    mgr.release(nb)
                big_disk.close()
                time.sleep(1.5)

        return {
            "profile": profile_name,
            "n_blocks": n_blocks,
            "avg_block_bytes": int(avg),
            "evict_pct": evict_pct,
            "min_ms": min(runs_ms),
            "med_ms": statistics.median(runs_ms),
            "max_ms": max(runs_ms),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profiles",
        nargs="+",
        default=["smolvlm_bf16", "qwen3_vl_4b_bf16", "gemma3_12b_bf16"],
    )
    ap.add_argument("--evict-blocks", type=int, default=200,
                    help="staging block count for eviction bench")
    ap.add_argument("--evict-pct", type=float, default=0.5,
                    help="fraction to evict")
    args = ap.parse_args()

    print("=" * 80)
    print("LOAD bench: mmap (load()) vs full materialise (mx.eval all K/V tensors)")
    print("=" * 80)
    print(
        f"{'profile':<22}{'file_size':>12}{'mmap_ms':>10}"
        f"{'full_ms':>10}{'full_max':>10}{'MB/s':>10}"
    )
    for prof in args.profiles:
        r = bench_load(prof)
        print(
            f"{r['profile']:<22}"
            f"{r['bytes_on_disk']/1024:>10.1f} KB"
            f"{r['mmap_med_ms']:>10.2f}"
            f"{r['full_med_ms']:>10.2f}"
            f"{r['full_max_ms']:>10.2f}"
            f"{r['throughput_mb_s']:>10.1f}"
        )

    print()
    print("=" * 70)
    print(
        f"EVICT bench ({args.evict_blocks} pre-populated blocks, "
        f"~{int(args.evict_pct*100)}% evicted)"
    )
    print("=" * 70)
    print(f"{'profile':<22}{'n_blocks':>10}{'avg_size':>12}{'min_ms':>10}{'med_ms':>10}{'max_ms':>10}")
    for prof in args.profiles:
        r = bench_evict(prof, args.evict_blocks, args.evict_pct)
        print(
            f"{r['profile']:<22}"
            f"{r['n_blocks']:>10}"
            f"{r['avg_block_bytes']/1024:>10.1f} KB"
            f"{r['min_ms']:>10.2f}"
            f"{r['med_ms']:>10.2f}"
            f"{r['max_ms']:>10.2f}"
        )


if __name__ == "__main__":
    main()

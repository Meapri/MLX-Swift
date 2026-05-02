"""End-to-end APC disk-tier bench: fill ~1000 blocks, then measure TTFT
cold / warm-from-disk / warm-from-memory plus full disk-side counters.

For each model:

    Step A — cold baseline
        Empty server, empty disk. Send test prompt once.
        Measures TTFT_cold (no APC help).

    Step B — fill phase
        Same server, send N varied prompts that each create unique blocks
        until we reach ≥1000 stored blocks. Cap at ~80% of total to force
        eviction of oldest blocks.

    Step C — restart & warm rounds
        Kill server (in-memory pool dies). Disk persists. Restart.
        Round 1: send test prompt — should hit disk (warm-from-disk).
        Round 2: send test prompt — should hit memory (warm-from-memory).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


SYSTEM = (
    "You are a careful technical writer producing detailed reference "
    "material. Always reply in full prose paragraphs without bullets."
)


def _make_unique_filler(idx: int, target_words: int) -> str:
    """Make a prompt whose first ~target_words words contain enough unique
    tokens that several full APC blocks diverge from any other prompt."""
    base = (
        f"unique-tag-{idx:05d} alpha bravo charlie delta echo foxtrot golf "
        f"hotel india juliet kilo lima mike november oscar papa quebec "
    )
    # repeat until long enough
    out = base
    while len(out.split()) < target_words:
        out += f" tag{idx:05d}-{len(out)} "
    return out


def chat(client, base, system, user, max_tokens=4, model: str = ""):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    r = client.post(
        f"{base}/v1/chat/completions", json=payload, timeout=300.0
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    return data, elapsed


def stream_ttft(client, base, system, user, max_tokens=2, model: str = ""):
    """TTFT measurement via SSE — time from POST to first content delta."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = None
    prompt_tokens = None
    with client.stream(
        "POST", f"{base}/v1/chat/completions", json=payload, timeout=300.0
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data in ("", "[DONE]"):
                continue
            obj = json.loads(data)
            choice = obj.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            if delta.get("content") and t_first is None:
                t_first = time.perf_counter()
            usage = obj.get("usage")
            if usage and prompt_tokens is None:
                prompt_tokens = usage.get("prompt_tokens")
    total = time.perf_counter() - t0
    return {
        "ttft_ms": (t_first - t0) * 1000.0 if t_first else float("inf"),
        "total_ms": total * 1000.0,
        "prompt_tokens": prompt_tokens,
    }


def stats(client, base):
    return client.get(f"{base}/v1/cache/stats", timeout=10.0).json()


def reset_stats(client, base):
    client.post(f"{base}/v1/cache/reset", timeout=10.0)


def start_server(model, port, disk_path: Path | None, disk_max_gb: float | None,
                 num_blocks: int = 4096) -> subprocess.Popen:
    env = os.environ.copy()
    env["APC_ENABLED"] = "1"
    env["APC_NUM_BLOCKS"] = str(num_blocks)
    if disk_path is not None:
        env["APC_DISK_PATH"] = str(disk_path)
    if disk_max_gb is not None:
        env["APC_DISK_MAX_GB"] = str(disk_max_gb)
    env["MLX_VLM_PRELOAD_MODEL"] = model
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_vlm.server",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 600.0  # gemma 12B can be slow to load
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=2.0).status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("server died during startup")
        time.sleep(2.0)
    proc.kill()
    raise RuntimeError("server didn't come up in 600s")


def stop_server(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
        proc.wait()


# Test prompt seed; ``_make_test_user`` extends it to roughly N tokens
# so a meaningful fraction of TTFT is genuine prefill cost (and thus
# something the disk tier can actually save).
_TEST_PREFIX = (
    "Explain in detail how a stack-based virtual machine differs from a "
    "register-based one. Cover instruction encoding density, decode "
    "complexity, register allocation strategy, exception handling, JIT "
    "considerations, and historical context for JVM, CLR, Lua, and Wasm. "
)


def _make_test_user(target_tokens: int) -> str:
    # ~3.3 tokens per word for typical English; pad until long enough.
    out = _TEST_PREFIX
    pad_words = [
        "Discuss memory model implications.",
        "Compare to dataflow VMs.",
        "Address peephole optimisation differences.",
        "Note implementation in compilers like LLVM.",
        "Mention how the Lua VM evolved across versions.",
        "Describe instruction selection on a register-based target.",
    ]
    i = 0
    while len(out.split()) * 3.3 < target_tokens:
        out += pad_words[i % len(pad_words)] + " "
        i += 1
    return out


def run_for_model(model: str, port: int, target_blocks: int, fill_prompts: int,
                  disk_max_gb: float, no_apc: bool = False,
                  test_prompt_tokens: int = 500) -> dict:
    print(f"\n{'=' * 70}\nModel: {model}\n{'=' * 70}")
    test_user = _make_test_user(test_prompt_tokens)

    disk_root = Path(tempfile.mkdtemp(prefix="apc-bench1k-"))
    base = f"http://127.0.0.1:{port}"
    result: dict = {"model": model, "no_apc": no_apc}

    try:
        # ---- Step A: cold baseline (no disk) ----
        print(f"  [A] cold baseline (empty server, no disk)…")
        proc = start_server(model, port, disk_path=None, disk_max_gb=None)
        try:
            with httpx.Client() as c:
                # Warm the JIT/MLX graph with one tiny prompt
                stream_ttft(c, base, "warmup", "hi", max_tokens=1, model=model)
                # Cold measurement
                r = stream_ttft(c, base, SYSTEM, test_user, max_tokens=2, model=model)
                print(f"      cold TTFT: {r['ttft_ms']:.0f}ms  "
                      f"prompt_tokens: {r['prompt_tokens']}")
                result["cold_ttft_ms"] = r["ttft_ms"]
                result["cold_prompt_tokens"] = r["prompt_tokens"]
                cold_prompt_tps = (r["prompt_tokens"] /
                                   ((r["ttft_ms"] / 1000.0) or 1.0))
                result["cold_prompt_tps"] = cold_prompt_tps
                print(f"      cold prompt_tps ≈ {cold_prompt_tps:.0f}")
        finally:
            stop_server(proc)

        if no_apc:
            return result

        # ---- Step B: fill phase ----
        print(f"  [B] fill {fill_prompts} prompts → ~{target_blocks} blocks "
              f"(disk cap {disk_max_gb} GB)…")
        proc = start_server(model, port, disk_path=disk_root,
                            disk_max_gb=disk_max_gb)
        try:
            with httpx.Client() as c:
                # Generate fill prompts
                t0 = time.perf_counter()
                for i in range(fill_prompts):
                    user = _make_unique_filler(i, target_words=160)
                    chat(c, base, SYSTEM, user, max_tokens=1, model=model)
                fill_elapsed = time.perf_counter() - t0
                fill_stats = stats(c, base)
                print(f"      fill done in {fill_elapsed:.1f}s")
                print(f"      stores={fill_stats['stores']} "
                      f"disk_writes={fill_stats.get('disk_writes', 0)} "
                      f"evictions={fill_stats['evictions']} "
                      f"disk_bytes={fill_stats.get('disk_bytes', 0)/1e6:.1f}MB")
                result["fill_stats"] = fill_stats
                result["fill_elapsed_s"] = fill_elapsed

                # Now send the test prompt once to STORE it on disk too,
                # so the warm rounds can find it on the freshly-restarted
                # server. (Without this, the test prompt's blocks would
                # never have hit disk.)
                stream_ttft(c, base, SYSTEM, test_user, max_tokens=1, model=model)
        finally:
            stop_server(proc)

        # Count files on disk now
        ns_dirs = [d for d in disk_root.iterdir() if d.is_dir()]
        if ns_dirs:
            files = [f for f in ns_dirs[0].glob("*.safetensors")
                     if len(f.stem) == 16]
            disk_size = sum(f.stat().st_size for f in files)
            result["files_on_disk"] = len(files)
            result["bytes_on_disk"] = disk_size
            print(f"      files on disk: {len(files)}, "
                  f"size: {disk_size/1e6:.1f}MB")

        time.sleep(1.5)

        # ---- Step C: restart, warm rounds ----
        print(f"  [C] restart + 2 warm rounds…")
        proc = start_server(model, port, disk_path=disk_root,
                            disk_max_gb=disk_max_gb)
        try:
            with httpx.Client() as c:
                # Tiny prompt to warm JIT before measuring
                stream_ttft(c, base, "warmup", "hi", max_tokens=1, model=model)
                reset_stats(c, base)

                # Round 1: should hit disk
                r1 = stream_ttft(c, base, SYSTEM, test_user, max_tokens=2, model=model)
                s1 = stats(c, base)
                # Round 2: should hit in-memory now
                r2 = stream_ttft(c, base, SYSTEM, test_user, max_tokens=2, model=model)
                s2 = stats(c, base)

                w1_tps = r1["prompt_tokens"] / ((r1["ttft_ms"] / 1000.0) or 1.0)
                w2_tps = r2["prompt_tokens"] / ((r2["ttft_ms"] / 1000.0) or 1.0)

                print(f"      warm-1 TTFT: {r1['ttft_ms']:.0f}ms  "
                      f"prompt_tokens: {r1['prompt_tokens']}  "
                      f"prompt_tps≈{w1_tps:.0f}")
                print(f"        stats: hits={s1['lookups_hit']} "
                      f"disk_hits={s1.get('disk_hits',0)} "
                      f"matched_tokens={s1['matched_tokens']}")
                print(f"      warm-2 TTFT: {r2['ttft_ms']:.0f}ms  "
                      f"prompt_tokens: {r2['prompt_tokens']}  "
                      f"prompt_tps≈{w2_tps:.0f}")
                print(f"        stats: hits={s2['lookups_hit']} "
                      f"disk_hits={s2.get('disk_hits',0)} "
                      f"matched_tokens={s2['matched_tokens']}")

                result["warm1_ttft_ms"] = r1["ttft_ms"]
                result["warm1_prompt_tps"] = w1_tps
                result["warm1_disk_hits"] = s1.get("disk_hits", 0)
                result["warm1_matched_tokens"] = s1["matched_tokens"]
                result["warm2_ttft_ms"] = r2["ttft_ms"]
                result["warm2_prompt_tps"] = w2_tps
                result["warm2_disk_hits"] = s2.get("disk_hits", 0)
                result["warm2_matched_tokens"] = s2["matched_tokens"]
        finally:
            stop_server(proc)
    finally:
        shutil.rmtree(disk_root, ignore_errors=True)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "mlx-community/SmolVLM-Instruct-bf16",
            "Qwen/Qwen3-VL-4B-Instruct",
            "mlx-community/gemma-3-12b-it-qat-bf16",
        ],
    )
    ap.add_argument("--target-blocks", type=int, default=1000)
    ap.add_argument("--fill-prompts", type=int, default=120,
                    help="number of varied prompts to submit during fill")
    ap.add_argument("--disk-cap-gb", type=float, default=2.0,
                    help="set lower than full to force eviction")
    ap.add_argument("--port-base", type=int, default=8089)
    ap.add_argument("--test-prompt-tokens", type=int, default=500,
                    help="size of the test prompt — longer surfaces "
                         "more prefill savings from disk-restore")
    args = ap.parse_args()

    # gemma3 has custom make_cache (Mamba-hybrid), so APC opts out — only
    # cold baseline is meaningful. Detect that pattern by model name.
    NO_APC = ("gemma-3", "gemma3", "nemotron-3-nano-omni")

    results = []
    for i, model in enumerate(args.models):
        no_apc = any(s in model.lower() for s in NO_APC)
        try:
            r = run_for_model(model, args.port_base + i,
                              args.target_blocks, args.fill_prompts,
                              args.disk_cap_gb, no_apc=no_apc,
                              test_prompt_tokens=args.test_prompt_tokens)
            results.append(r)
        except Exception as e:
            print(f"  ✗ {model} bench failed: {e}")
            results.append({"model": model, "error": str(e)})

    # Final report
    print("\n" + "=" * 110)
    print(f"Summary (test prompt ≈ {args.test_prompt_tokens} tokens)")
    print("=" * 110)
    print(
        f"{'model':<32}"
        f"{'cold (ttft / tps)':>22}"
        f"{'warm-disk (ttft / tps)':>26}"
        f"{'warm-mem (ttft / tps)':>26}"
    )
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<30} ERROR: {r['error']}")
            continue
        if r.get("no_apc"):
            print(
                f"  {r['model']:<30}"
                f"{r['cold_ttft_ms']:>9.0f}ms /{r['cold_prompt_tps']:>6.0f} tps"
                "  (APC opts out: model uses custom KVCache)"
            )
            continue
        print(
            f"  {r['model']:<30}"
            f"{r['cold_ttft_ms']:>9.0f}ms /{r['cold_prompt_tps']:>6.0f} tps"
            f"{r['warm1_ttft_ms']:>13.0f}ms /{r['warm1_prompt_tps']:>6.0f} tps"
            f"{r['warm2_ttft_ms']:>13.0f}ms /{r['warm2_prompt_tps']:>6.0f} tps"
        )

    # Per-model detail block
    print()
    print("Detail (fill / disk / matches):")
    print("-" * 110)
    for r in results:
        if "error" in r or r.get("no_apc"):
            continue
        f = r.get("fill_stats", {}) or {}
        ttft_save_disk = r["cold_ttft_ms"] - r["warm1_ttft_ms"]
        ttft_save_mem = r["cold_ttft_ms"] - r["warm2_ttft_ms"]
        prefill_save_disk = (
            r["cold_prompt_tokens"] / max(r["cold_prompt_tps"], 1) -
            r["cold_prompt_tokens"] / max(r["warm1_prompt_tps"], 1)
        ) * 1000.0
        print(
            f"  {r['model']}\n"
            f"    fill: prompts={args.fill_prompts} stores={f.get('stores',0)} "
            f"mem_evict={f.get('evictions',0)} disk_writes={f.get('disk_writes',0)}\n"
            f"    disk: cap={args.disk_cap_gb:.1f}GB usage={f.get('disk_bytes',0)/1e6:.0f}MB "
            f"files={f.get('disk_files','?')} disk_evict={f.get('disk_evictions','?')}\n"
            f"    warm1: disk_hits={r['warm1_disk_hits']} "
            f"matched={r['warm1_matched_tokens']} TTFT_saved={ttft_save_disk:.0f}ms "
            f"({ttft_save_disk / r['cold_ttft_ms'] * 100:.0f}% of cold)\n"
            f"    warm2: disk_hits={r['warm2_disk_hits']} "
            f"matched={r['warm2_matched_tokens'] - r['warm1_matched_tokens']} "
            f"TTFT_saved={ttft_save_mem:.0f}ms "
            f"({ttft_save_mem / r['cold_ttft_ms'] * 100:.0f}% of cold)"
        )

    print("\nDetailed per-model:\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Compare TTFT for: (a) cold server, no disk; (b) cold server, warm disk."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import json


PROMPT = (
    "You are a careful technical writer producing reference material. "
    "Always reply with a multi-paragraph explanation: at least three paragraphs, "
    "with concrete examples and historical context where applicable. "
    "Avoid bullet points; use plain prose only. "
    "Explain how a stack-based VM differs from a register-based one in detail."
)


def start_server(model: str, port: int, disk_path: Path | None) -> subprocess.Popen:
    env = os.environ.copy()
    env["APC_ENABLED"] = "1"
    env["APC_NUM_BLOCKS"] = "1024"
    if disk_path is not None:
        env["APC_DISK_PATH"] = str(disk_path)
    env["MLX_VLM_PRELOAD_MODEL"] = model
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_vlm.server", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120.0
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=2.0)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("server died during startup")
        time.sleep(1.0)
    proc.kill()
    raise RuntimeError("server didn't come up")


def stop_server(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait()


def stream_ttft(client, base, model, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t_post = time.perf_counter()
    t_first = None
    n = 0
    with client.stream("POST", f"{base}/v1/chat/completions", json=payload, timeout=120.0) as r:
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
            content = delta.get("content")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                n += 1
    return (t_first - t_post) * 1000.0, (time.perf_counter() - t_post) * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--max-tokens", type=int, default=4)
    args = p.parse_args()

    disk_root = Path(tempfile.mkdtemp(prefix="apc-disk-"))
    print(f"disk root: {disk_root}")

    try:
        # ---- Pass A: server with disk path, send request to populate disk ----
        print("\n[pass A] cold server + cold disk (populate)…")
        proc = start_server(args.model, args.port, disk_root)
        base = f"http://127.0.0.1:{args.port}"
        try:
            with httpx.Client() as c:
                ttft_a, total_a = stream_ttft(c, base, args.model, args.max_tokens)
                print(f"  TTFT={ttft_a:.0f}ms total={total_a:.0f}ms (cold disk; populates)")
                s = c.get(f"{base}/v1/cache/stats", timeout=5.0).json()
                print(f"  stats: stores={s['stores']} disk_writes={s.get('disk_writes')}")
        finally:
            stop_server(proc)

        time.sleep(1.0)

        # ---- Pass B: restart, same disk, expect disk hits ----
        print("\n[pass B] cold server + warm disk (restored from disk)…")
        proc = start_server(args.model, args.port, disk_root)
        try:
            with httpx.Client() as c:
                ttft_b, total_b = stream_ttft(c, base, args.model, args.max_tokens)
                print(f"  TTFT={ttft_b:.0f}ms total={total_b:.0f}ms (warm disk)")
                s = c.get(f"{base}/v1/cache/stats", timeout=5.0).json()
                print(
                    f"  stats: hits={s['lookups_hit']} disk_hits={s.get('disk_hits')} "
                    f"matched_tokens={s['matched_tokens']}"
                )
        finally:
            stop_server(proc)

        time.sleep(1.0)

        # ---- Pass C: cold server, NO disk path (control) ----
        print("\n[pass C] cold server + no disk (control)…")
        proc = start_server(args.model, args.port, None)
        try:
            with httpx.Client() as c:
                ttft_c, total_c = stream_ttft(c, base, args.model, args.max_tokens)
                print(f"  TTFT={ttft_c:.0f}ms total={total_c:.0f}ms (no-disk baseline)")
        finally:
            stop_server(proc)

        print("\n" + "=" * 50)
        print(f"cold (populates disk) TTFT:    {ttft_a:.0f}ms")
        print(f"warm-from-disk TTFT:           {ttft_b:.0f}ms")
        print(f"no-disk baseline TTFT:         {ttft_c:.0f}ms")
        if ttft_b < ttft_c:
            print(f"\nDisk-restore wins {ttft_c - ttft_b:.0f}ms vs cold prefill")
        else:
            print(f"\nNo win; disk overhead {ttft_b - ttft_c:.0f}ms")
    finally:
        shutil.rmtree(disk_root, ignore_errors=True)


if __name__ == "__main__":
    main()

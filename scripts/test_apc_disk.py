"""APC disk-tier round-trip via the FastAPI server.

Phase 1: start the server with APC_DISK_PATH set; fire a request. Verify
the cache file landed on disk.

Phase 2: restart the server (same APC_DISK_PATH); fire the same request.
Verify the cache stats report ``disk_hits >= 1``, meaning the prefix was
restored from SSD instead of re-prefilled.
"""

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


PROMPT = (
    "You are a careful technical writer producing reference material. "
    "Always reply with a multi-paragraph explanation: at least three paragraphs, "
    "with concrete examples and historical context where applicable. "
    "Avoid bullet points; use plain prose only. "
    "Explain how a stack-based VM differs from a register-based one."
)


def chat(client, base, model, max_tokens=8):
    r = client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()


def stats(client, base):
    return client.get(f"{base}/v1/cache/stats", timeout=5.0).json()


def reset(client, base):
    client.post(f"{base}/v1/cache/reset", timeout=5.0)


def start_server(model: str, port: int, disk_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["APC_ENABLED"] = "1"
    env["APC_NUM_BLOCKS"] = "1024"
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
    raise RuntimeError("server didn't come up in 120s")


def stop_server(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--keep-disk", action="store_true", help="don't delete the disk dir at the end")
    args = p.parse_args()

    disk_root = Path(tempfile.mkdtemp(prefix="apc-disk-"))
    print(f"disk root: {disk_root}")

    failures = []
    try:
        # ---- Phase 1: cold server, store on disk ----
        print("\n[phase 1] starting server (cold disk)…")
        proc = start_server(args.model, args.port, disk_root)
        base = f"http://127.0.0.1:{args.port}"
        try:
            with httpx.Client() as c:
                reset(c, base)
                chat(c, base, args.model, args.max_tokens)
                s1 = stats(c, base)
                print(f"  after request: stores={s1['stores']} disk_writes={s1.get('disk_writes', '?')}")
                if s1.get("disk_writes", 0) <= 0:
                    failures.append("phase 1 should have written to disk")
        finally:
            stop_server(proc)

        # Disk should now have at least one .safetensors file
        ns_dir = disk_root / [d.name for d in disk_root.iterdir() if d.is_dir()][0]
        files = list(ns_dir.glob("*.safetensors"))
        files = [f for f in files if f.stem.startswith("shard_")]
        print(f"  files on disk: {len(files)}")
        if not files:
            failures.append("expected at least one .safetensors file on disk")

        # Pause so the OS-level write definitely flushes.
        time.sleep(1.0)

        # ---- Phase 2: restart, expect disk hits ----
        print("\n[phase 2] restarting server (warm disk)…")
        proc = start_server(args.model, args.port, disk_root)
        try:
            with httpx.Client() as c:
                # No reset! The disk-loaded blocks should be discoverable.
                chat(c, base, args.model, args.max_tokens)
                s2 = stats(c, base)
                print(
                    f"  after request: hits={s2['lookups_hit']} disk_hits={s2.get('disk_hits', '?')} "
                    f"matched_tokens={s2['matched_tokens']} stores={s2['stores']}"
                )
                if s2.get("disk_hits", 0) <= 0:
                    failures.append(
                        "phase 2 should have promoted at least one block from disk"
                    )
                if s2["matched_tokens"] <= 0:
                    failures.append(
                        "phase 2 should have matched at least one prefix block"
                    )
        finally:
            stop_server(proc)
    finally:
        if not args.keep_disk:
            shutil.rmtree(disk_root, ignore_errors=True)
        else:
            print(f"\n(kept disk dir: {disk_root})")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("All disk-persistence assertions passed.")


if __name__ == "__main__":
    main()

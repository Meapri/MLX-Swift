"""Staggered concurrent APC test.

Fires N requests with a configurable inter-arrival delay between launches
so they hit the server *while a previous one is mid-decode*. This stresses
the case where new arrivals must extend an already-decoding batch.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx


SHARED_SYSTEM = (
    "You are a careful technical writer producing reference material. "
    "Always reply with a multi-paragraph explanation: at least three paragraphs, "
    "with concrete examples and historical context where applicable. "
    "Avoid bullet points; use plain prose only."
)
USERS = [
    "Explain how a stack-based VM differs from a register-based one.",
    "Describe the role of merkle trees in modern version control.",
    "Walk through how DNS resolution works for a typical web request.",
    "Explain reference counting versus tracing garbage collection.",
]


async def fire(client, url, model, system, user, idx, max_tokens):
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
    r = await client.post(url, json=payload, timeout=120.0)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    return idx, data["choices"][0]["message"]["content"], elapsed, t0


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8089")
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--stagger-ms",
        type=int,
        default=200,
        help="Delay between successive request launches",
    )
    args = p.parse_args()

    url = f"{args.base}/v1/chat/completions"

    async with httpx.AsyncClient() as client:
        # Reset stats + warm-up so the system prompt is cached.
        await client.post(f"{args.base}/v1/cache/reset", timeout=10.0)
        print("Warm-up…")
        await fire(client, url, args.model, SHARED_SYSTEM, USERS[0], -1, args.max_tokens)

        await client.post(f"{args.base}/v1/cache/reset", timeout=10.0)

        print(
            f"\nFiring {args.n} requests with {args.stagger_ms}ms stagger "
            f"(max_tokens={args.max_tokens})…"
        )
        run_t0 = time.perf_counter()
        tasks = []
        for i in range(args.n):
            user = USERS[i % len(USERS)]
            tasks.append(
                asyncio.create_task(
                    fire(client, url, args.model, SHARED_SYSTEM, user, i, args.max_tokens)
                )
            )
            await asyncio.sleep(args.stagger_ms / 1000.0)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - run_t0

        print(f"  total wall-clock (incl. stagger): {wall:.2f}s")
        for r in results:
            if isinstance(r, Exception):
                print(f"  ✗ {r!r}")
                sys.exit(1)
            idx, text, elapsed, launched = r
            launch_off = launched - run_t0
            print(
                f"  [{idx}] launched=+{launch_off:.2f}s elapsed={elapsed:.2f}s "
                f"text={text[:80]!r}"
            )

        s = (await client.get(f"{args.base}/v1/cache/stats", timeout=5.0)).json()
        print(
            f"\nAPC: hits={s['lookups_hit']} miss={s['lookups_miss']} "
            f"matched_tokens={s['matched_tokens']} stores={s['stores']} "
            f"hit_rate={s['token_hit_rate']:.1%}"
        )


if __name__ == "__main__":
    asyncio.run(main())

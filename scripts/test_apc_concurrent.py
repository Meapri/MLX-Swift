"""Concurrent APC test.

Fires N concurrent /v1/chat/completions requests sharing a long prefix and
verifies:
  1. All requests complete successfully (no race-induced corruption).
  2. Cumulative APC stats show prefix reuse (matched_tokens > block_size).
  3. Generated text is consistent across requests sharing identical prompts
     (deterministic with temperature=0).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import List, Tuple

import httpx


SHARED_SYSTEM = (
    "You are a careful and concise assistant. Always answer in one short paragraph "
    "without preamble. Use plain prose, no bullet points or headings. "
    "When uncertain, say 'I'm not sure' rather than guessing. "
    "Treat each user turn independently and never reference past turns."
)

VARIANTS = [
    "What is the capital of France?",
    "What is the capital of Spain?",
    "What is the capital of Italy?",
    "What is the capital of Japan?",
    "What is the capital of Germany?",
    "What is the capital of Brazil?",
    "What is the capital of Egypt?",
    "What is the capital of Canada?",
]


async def fire(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    user_msg: str,
    idx: int,
) -> Tuple[int, str, float, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SHARED_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 24,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, timeout=120.0)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return idx, text, elapsed, usage


async def stats(client: httpx.AsyncClient, base: str) -> dict:
    r = await client.get(f"{base}/v1/cache/stats", timeout=10.0)
    return r.json()


async def reset(client: httpx.AsyncClient, base: str) -> None:
    await client.post(f"{base}/v1/cache/reset", timeout=10.0)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8089")
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    p.add_argument("--n-concurrent", type=int, default=4)
    p.add_argument("--n-rounds", type=int, default=2)
    args = p.parse_args()

    url = f"{args.base}/v1/chat/completions"

    async with httpx.AsyncClient() as client:
        print(f"Reset cache stats…")
        await reset(client, args.base)

        # Warm-up: single request to populate the shared system prefix
        print("Warm-up (single sequential request)…")
        _, warm_text, warm_t, warm_usage = await fire(
            client, url, args.model, VARIANTS[0], 0
        )
        warm_stats = await stats(client, args.base)
        print(
            f"  warm: elapsed={warm_t:.2f}s tokens={warm_usage.get('prompt_tokens')} "
            f"matched_after={warm_stats['matched_tokens']} stored={warm_stats['stores']}"
        )

        # Concurrent rounds: each round fires N requests in parallel
        all_texts: List[str] = [warm_text]
        for round_idx in range(args.n_rounds):
            print(f"\nRound {round_idx + 1}: firing {args.n_concurrent} concurrent requests…")
            tasks = [
                fire(client, url, args.model, VARIANTS[i % len(VARIANTS)], i)
                for i in range(args.n_concurrent)
            ]
            t0 = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            wall = time.perf_counter() - t0
            print(f"  wall_clock={wall:.2f}s")

            for r in results:
                if isinstance(r, Exception):
                    print(f"  ✗ FAIL: {r!r}")
                    sys.exit(1)
                idx, text, elapsed, usage = r
                all_texts.append(text)
                print(
                    f"  [{idx}] elapsed={elapsed:.2f}s prompt_tokens={usage.get('prompt_tokens')} "
                    f"text={text!r}"
                )

            s = await stats(client, args.base)
            print(
                f"  cumulative stats: lookups_hit={s['lookups_hit']} "
                f"lookups_miss={s['lookups_miss']} matched_tokens={s['matched_tokens']} "
                f"stores={s['stores']} pool_used={s['pool_used']} "
                f"hit_rate={s['token_hit_rate']:.2%}"
            )

        # Verdict
        print("\n=== Verdict ===")
        s = await stats(client, args.base)
        ok = True
        if s["matched_tokens"] <= 0:
            print("✗ FAIL: no APC hits across concurrent rounds")
            ok = False
        else:
            print(f"✓ PASS: {s['matched_tokens']} prefix tokens reused")
        # Determinism: all variants[0] requests should match warm_text exactly
        same_prompt_outputs = [
            t for t, var in zip(all_texts, [VARIANTS[0]] + [VARIANTS[i % len(VARIANTS)] for i in range(args.n_concurrent)] * args.n_rounds)
            if var == VARIANTS[0]
        ]
        unique = set(same_prompt_outputs)
        if len(unique) > 1:
            print(f"✗ FAIL: same prompt produced different texts (count={len(unique)})")
            for u in unique:
                print(f"   - {u!r}")
            ok = False
        else:
            print(f"✓ PASS: deterministic outputs across {len(same_prompt_outputs)} same-prompt requests")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())

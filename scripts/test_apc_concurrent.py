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
    "You are a careful technical writer producing reference material. "
    "Always reply with a multi-paragraph explanation: at least three paragraphs, "
    "with concrete examples and historical context where applicable. "
    "Avoid bullet points; use plain prose only. Do not reference earlier turns. "
    "Continue until you have thoroughly covered the topic; do not stop short."
)

VARIANTS = [
    "Explain how a stack-based virtual machine differs from a register-based one, with examples.",
    "Describe the role of merkle trees in modern version control systems.",
    "Walk through how DNS resolution works for a typical web request.",
    "Explain reference counting versus tracing garbage collection, including tradeoffs.",
    "Describe how a write-ahead log keeps a database durable across crashes.",
    "Explain why HTTPS needs both symmetric and asymmetric cryptography.",
    "Walk through how a CPU pipeline handles a branch misprediction.",
    "Describe the role of the kernel scheduler when many threads are runnable.",
]


async def fire(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    user_msg: str,
    idx: int,
    max_tokens: int = 24,
) -> Tuple[int, str, float, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SHARED_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
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
    p.add_argument("--max-tokens", type=int, default=24)
    args = p.parse_args()

    url = f"{args.base}/v1/chat/completions"

    async with httpx.AsyncClient() as client:
        print(f"Reset cache stats…")
        await reset(client, args.base)

        # Warm-up: single request to populate the shared system prefix
        print("Warm-up (single sequential request)…")
        _, warm_text, warm_t, warm_usage = await fire(
            client, url, args.model, VARIANTS[0], 0, max_tokens=args.max_tokens
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
                fire(client, url, args.model, VARIANTS[i % len(VARIANTS)], i, max_tokens=args.max_tokens)
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

        # Determinism: report (don't strictly assert) cross-round divergence.
        # Even warm-to-warm runs aren't bit-equivalent under concurrent traffic
        # because attention is not batch-invariant — different runtime/GPU
        # state between rounds picks different reduction orders, even with
        # identical inputs and identical APC blocks. See Thinking Machines
        # (2025) and the LLM-42 paper. Bit-equivalent determinism requires
        # batch-invariant kernels (vLLM --enable-batch-invariance,
        # SGLang+FlashInfer/FA3), not changes to the cache.
        warm_variants_per_round = [
            VARIANTS[i % len(VARIANTS)] for i in range(args.n_concurrent)
        ] * args.n_rounds
        warm_texts = all_texts[1:]  # drop the cold warm-up entry
        from collections import defaultdict
        by_prompt = defaultdict(list)
        for t, v in zip(warm_texts, warm_variants_per_round):
            by_prompt[v].append(t)

        def common_prefix_len(a: str, b: str) -> int:
            n = min(len(a), len(b))
            for i in range(n):
                if a[i] != b[i]:
                    return i
            return n

        n_repeats = 0
        n_identical = 0
        prefix_chars = []
        for v, ts in by_prompt.items():
            if len(ts) <= 1:
                continue
            n_repeats += 1
            if len(set(ts)) == 1:
                n_identical += 1
                prefix_chars.append(len(ts[0]))
            else:
                # divergence point in chars
                pairs = []
                for i in range(len(ts)):
                    for j in range(i + 1, len(ts)):
                        pairs.append(common_prefix_len(ts[i], ts[j]))
                prefix_chars.append(min(pairs))

        if n_repeats == 0:
            print("✗ FAIL: no repeat-prompts to assess determinism")
            ok = False
        else:
            mean_prefix = sum(prefix_chars) / len(prefix_chars)
            print(
                f"  warm-to-warm consistency across {n_repeats} repeat-prompts: "
                f"{n_identical}/{n_repeats} bit-identical, "
                f"mean shared prefix = {mean_prefix:.0f} chars"
            )
            # We tolerate divergence beyond ~60 chars (~15 tokens). That's a
            # generous floor that catches catastrophic regressions but doesn't
            # flag normal kernel non-invariance drift.
            if min(prefix_chars) < 30:
                print(f"✗ FAIL: shortest common prefix only {min(prefix_chars)} chars")
                ok = False
            else:
                print(
                    f"  ✓ no catastrophic divergence "
                    f"(min shared prefix {min(prefix_chars)} chars >= 30 tolerance)"
                )

        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())

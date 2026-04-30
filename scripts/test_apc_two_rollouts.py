"""Two-prompt APC rollout test.

Runs two distinct prompt rollouts (A and B), interleaving them, and verifies:
  1. First time we see prompt A → no hit, blocks stored.
  2. Second time we see prompt A → cache hit (prefix tokens reused).
  3. Prompt B (totally different prefix) → no hit, stores its own blocks.
  4. Second time we see prompt B → cache hit.
  5. Prompts don't cross-contaminate (deterministic, distinct outputs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


PROMPT_A_SYSTEM = (
    "You are a senior Rust engineer. Reply with concise, idiomatic Rust code "
    "and a brief one-sentence explanation underneath. Use stable Rust 1.85+ "
    "features only. Do not include markdown fences."
)
PROMPT_A_USER = "Implement an iterator that yields the running max of an i64 stream."

PROMPT_B_SYSTEM = (
    "You are a marine biologist. Answer in three short sentences without bullets. "
    "Always include the scientific name of any animal you mention."
)
PROMPT_B_USER = "Why do octopuses have three hearts?"


def chat(client: httpx.Client, base: str, model: str, system: str, user: str):
    t0 = time.perf_counter()
    r = client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"], elapsed, data.get("usage", {})


def stats(client: httpx.Client, base: str) -> dict:
    return client.get(f"{base}/v1/cache/stats", timeout=10.0).json()


def reset(client: httpx.Client, base: str) -> None:
    client.post(f"{base}/v1/cache/reset", timeout=10.0)


def diff_stats(before: dict, after: dict) -> dict:
    return {
        k: after[k] - before[k]
        for k in ("lookups_hit", "lookups_miss", "matched_tokens", "stores", "evictions")
    }


def run_step(client, base, model, label, system, user):
    s_before = stats(client, base)
    text, elapsed, usage = chat(client, base, model, system, user)
    s_after = stats(client, base)
    delta = diff_stats(s_before, s_after)
    print(
        f"[{label}] elapsed={elapsed:.2f}s prompt_tokens={usage.get('prompt_tokens')} "
        f"Δhits={delta['lookups_hit']} Δmiss={delta['lookups_miss']} "
        f"Δmatched={delta['matched_tokens']} Δstores={delta['stores']} "
        f"text={text!r}"
    )
    return text, elapsed, delta, s_after


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8089")
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    args = p.parse_args()

    failures = []

    with httpx.Client() as client:
        reset(client, args.base)
        print("Cache reset.\n")

        # Round 1 — first time for both A and B
        print("=== Round 1 (cold) ===")
        a1_text, _, a1_delta, _ = run_step(
            client, args.base, args.model, "A1", PROMPT_A_SYSTEM, PROMPT_A_USER
        )
        b1_text, _, b1_delta, _ = run_step(
            client, args.base, args.model, "B1", PROMPT_B_SYSTEM, PROMPT_B_USER
        )

        if a1_delta["matched_tokens"] != 0:
            failures.append(f"A1 should be a cold miss; got matched={a1_delta['matched_tokens']}")
        if b1_delta["matched_tokens"] != 0:
            failures.append(f"B1 should be a cold miss; got matched={b1_delta['matched_tokens']}")
        if a1_delta["stores"] <= 0:
            failures.append("A1 should have stored at least 1 block")
        if b1_delta["stores"] <= 0:
            failures.append("B1 should have stored at least 1 block")

        # Round 2 — repeat both
        print("\n=== Round 2 (warm) ===")
        a2_text, _, a2_delta, _ = run_step(
            client, args.base, args.model, "A2", PROMPT_A_SYSTEM, PROMPT_A_USER
        )
        b2_text, _, b2_delta, _ = run_step(
            client, args.base, args.model, "B2", PROMPT_B_SYSTEM, PROMPT_B_USER
        )

        if a2_delta["matched_tokens"] <= 0:
            failures.append(f"A2 should hit cache; got matched={a2_delta['matched_tokens']}")
        if b2_delta["matched_tokens"] <= 0:
            failures.append(f"B2 should hit cache; got matched={b2_delta['matched_tokens']}")

        # Cross-contamination check: A and B answers should be distinct
        if a1_text == b1_text:
            failures.append("A and B produced identical text — cross-contamination?")

        # Round 3 — interleave again to confirm nothing was evicted
        print("\n=== Round 3 (interleaved warm) ===")
        a3_text, _, a3_delta, _ = run_step(
            client, args.base, args.model, "A3", PROMPT_A_SYSTEM, PROMPT_A_USER
        )
        b3_text, _, b3_delta, _ = run_step(
            client, args.base, args.model, "B3", PROMPT_B_SYSTEM, PROMPT_B_USER
        )

        if a3_delta["matched_tokens"] <= 0:
            failures.append("A3 should still hit after B intervened")
        if b3_delta["matched_tokens"] <= 0:
            failures.append("B3 should still hit after A intervened")

        # Note: cold-vs-warm outputs are NOT guaranteed bit-equivalent because
        # MLX's flash-attention kernel uses different tile shapes when the
        # query length changes (long Q for cold prefill vs short Q for warm).
        # K/V values at cached positions are byte-identical (verified in
        # scripts/debug_apc_drift.py) but the per-layer attention output
        # drifts ~0.1-0.3 in bf16, which can occasionally flip argmax at
        # near-tied tokens. vLLM and sglang have the same property. We
        # instead assert warm-to-warm self-consistency.
        if a2_text != a3_text:
            failures.append(
                f"A: warm-to-warm non-deterministic\n  a2={a2_text!r}\n  a3={a3_text!r}"
            )
        if b2_text != b3_text:
            failures.append(
                f"B: warm-to-warm non-deterministic\n  b2={b2_text!r}\n  b3={b3_text!r}"
            )

        final = stats(client, args.base)
        print()
        print("Final stats:", json.dumps(final, indent=2))

    print()
    print(f"Cold A: {a1_text!r}")
    print(f"Warm A: {a2_text!r}")
    print(f"Cold B: {b1_text!r}")
    print(f"Warm B: {b2_text!r}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()

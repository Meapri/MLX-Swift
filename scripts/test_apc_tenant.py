"""APC per-tenant salt isolation test.

Two clients send the *same* prompt with different ``X-APC-Tenant`` headers.
Tenant A's first call is cold and stores blocks. Tenant B's call must NOT
hit those blocks (its hash chain is salted differently). A second call from
tenant A must hit (its salt is the same).
"""

from __future__ import annotations

import argparse
import sys

import httpx

PROMPT = (
    "You are a confidential medical assistant. Answer concisely. "
    "What are common side effects of metformin?"
)


def chat(client, base, model, tenant=None, max_tokens=8):
    headers = {}
    if tenant is not None:
        headers["X-APC-Tenant"] = tenant
    r = client.post(
        f"{base}/v1/chat/completions",
        headers=headers,
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8089")
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()

    failures = []

    with httpx.Client() as c:
        reset(c, args.base)
        s0 = stats(c, args.base)

        # Tenant A — first call: cold miss, stores blocks.
        chat(c, args.base, args.model, tenant="alice", max_tokens=args.max_tokens)
        s1 = stats(c, args.base)
        d_a1 = {k: s1[k] - s0[k] for k in ("lookups_hit", "lookups_miss", "stores")}
        print(
            f"A1 (alice cold): Δhits={d_a1['lookups_hit']} Δmiss={d_a1['lookups_miss']} Δstores={d_a1['stores']}"
        )
        if d_a1["lookups_hit"] != 0 or d_a1["stores"] <= 0:
            failures.append(f"A1 should be cold + store blocks; got {d_a1}")

        # Tenant B — same prompt, different salt: must MISS.
        chat(c, args.base, args.model, tenant="bob", max_tokens=args.max_tokens)
        s2 = stats(c, args.base)
        d_b1 = {k: s2[k] - s1[k] for k in ("lookups_hit", "lookups_miss", "stores")}
        print(
            f"B1 (bob, same prompt): Δhits={d_b1['lookups_hit']} Δmiss={d_b1['lookups_miss']} Δstores={d_b1['stores']}"
        )
        if d_b1["lookups_hit"] != 0:
            failures.append(
                f"B1 leaked Alice's blocks! Δhits={d_b1['lookups_hit']} (expected 0)"
            )
        if d_b1["stores"] <= 0:
            failures.append("B1 should store its own blocks (different salt)")

        # Tenant A — same prompt again: must HIT.
        chat(c, args.base, args.model, tenant="alice", max_tokens=args.max_tokens)
        s3 = stats(c, args.base)
        d_a2 = {k: s3[k] - s2[k] for k in ("lookups_hit", "lookups_miss", "stores")}
        print(
            f"A2 (alice repeat): Δhits={d_a2['lookups_hit']} Δmiss={d_a2['lookups_miss']} Δstores={d_a2['stores']}"
        )
        if d_a2["lookups_hit"] != 1:
            failures.append(
                f"A2 should hit Alice's prior blocks; got Δhits={d_a2['lookups_hit']}"
            )

        # Tenant B — repeat: must HIT its own blocks.
        chat(c, args.base, args.model, tenant="bob", max_tokens=args.max_tokens)
        s4 = stats(c, args.base)
        d_b2 = {k: s4[k] - s3[k] for k in ("lookups_hit", "lookups_miss", "stores")}
        print(
            f"B2 (bob repeat): Δhits={d_b2['lookups_hit']} Δmiss={d_b2['lookups_miss']} Δstores={d_b2['stores']}"
        )
        if d_b2["lookups_hit"] != 1:
            failures.append(
                f"B2 should hit Bob's prior blocks; got Δhits={d_b2['lookups_hit']}"
            )

        # No-header request — uses unsalted bucket; must MISS both Alice & Bob.
        chat(c, args.base, args.model, tenant=None, max_tokens=args.max_tokens)
        s5 = stats(c, args.base)
        d_n = {k: s5[k] - s4[k] for k in ("lookups_hit", "lookups_miss", "stores")}
        print(
            f"N1 (no header): Δhits={d_n['lookups_hit']} Δmiss={d_n['lookups_miss']} Δstores={d_n['stores']}"
        )
        if d_n["lookups_hit"] != 0:
            failures.append(
                f"Anonymous request leaked tenant blocks! Δhits={d_n['lookups_hit']}"
            )

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("All tenant-isolation assertions passed.")


if __name__ == "__main__":
    main()

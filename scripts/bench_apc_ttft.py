"""Benchmark APC time-to-first-token.

Fires a streaming chat completion three times:
  1. Cold (cache reset → no hit)
  2. Warm (full prefix hit)
  3. Half-warm (small new suffix on the same prefix)

Measures TTFT (post → first SSE chunk with non-empty delta), prefill time,
and total wall-clock. Run against an already-running server.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

LONG_SYSTEM = (
    "You are a senior staff engineer reviewing pull requests. Always start by "
    "summarising the change in one sentence, then list concrete concerns by "
    "file:line, then end with a verdict of APPROVE / REQUEST_CHANGES / COMMENT. "
    "Never speculate beyond the diff. Never accept code that lacks tests for "
    "new public API surface. Be concise but specific. Avoid corporate hedging. "
    "If you do not understand the surrounding architecture, say so explicitly "
    "rather than guess. When suggesting fixes, prefer minimal patches over "
    "rewrites. Treat any flaky test as a bug, not noise. Quote the offending "
    "line verbatim before commenting on it. End every review with a one-line "
    "summary of the highest-priority blocker, or NONE if there is none."
)

USERS = [
    "Review this diff: function `parse(s)` now returns `Optional[int]` instead of raising.",
    "Review this diff: added a 60-second timeout to all outbound HTTP requests.",
    "Review this diff: replaced the in-memory dedup set with a Redis SET.",
]


def stream_ttft(
    client: httpx.Client, base: str, model: str, system: str, user: str, max_tokens: int
):
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
    t_post = time.perf_counter()
    t_first = None
    n_tokens = 0
    text = ""
    prompt_tokens = None

    with client.stream(
        "POST",
        f"{base}/v1/chat/completions",
        json=payload,
        timeout=120.0,
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
            content = delta.get("content")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                text += content
                n_tokens += 1
            usage = obj.get("usage")
            if usage and prompt_tokens is None:
                prompt_tokens = usage.get("prompt_tokens")
    t_done = time.perf_counter()
    return {
        "ttft_ms": (t_first - t_post) * 1000.0 if t_first else float("inf"),
        "total_ms": (t_done - t_post) * 1000.0,
        "tokens": n_tokens,
        "text": text,
        "prompt_tokens": prompt_tokens,
    }


def get_stats(client: httpx.Client, base: str) -> dict:
    return client.get(f"{base}/v1/cache/stats", timeout=5.0).json()


def reset_stats(client: httpx.Client, base: str) -> None:
    client.post(f"{base}/v1/cache/reset", timeout=5.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8089")
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument(
        "--n-warm", type=int, default=3, help="warm-call repeats for averaging"
    )
    args = p.parse_args()

    with httpx.Client() as c:
        # Sanity: server up?
        h = c.get(f"{args.base}/health", timeout=5.0).json()
        print("health:", h)

        if not h.get("apc_enabled"):
            print("APC is not enabled on this server.")
            sys.exit(2)

        # Discard the first request — it has cold-import overhead unrelated to APC.
        reset_stats(c, args.base)
        print("\nWarmup (cold) + Discard (excludes import & first-graph-build)…")
        warm0 = stream_ttft(
            c, args.base, args.model, LONG_SYSTEM, USERS[0], args.max_tokens
        )
        print(
            f"  discarded TTFT={warm0['ttft_ms']:.0f}ms total={warm0['total_ms']:.0f}ms"
        )

        # ---- Cold ----
        reset_stats(c, args.base)
        print("\nCOLD (cache reset before each call):")
        cold = []
        for i, u in enumerate(USERS):
            reset_stats(c, args.base)
            r = stream_ttft(c, args.base, args.model, LONG_SYSTEM, u, args.max_tokens)
            cold.append(r)
            print(
                f"  [{i}] TTFT={r['ttft_ms']:.0f}ms  total={r['total_ms']:.0f}ms  "
                f"prompt_tokens={r['prompt_tokens']}  text={r['text']!r}"
            )

        # ---- Warm ----
        # Warm the prefix first
        reset_stats(c, args.base)
        _ = stream_ttft(
            c, args.base, args.model, LONG_SYSTEM, USERS[0], args.max_tokens
        )
        s_after_seed = get_stats(c, args.base)
        print(
            f"\nSeeded prefix; pool_used={s_after_seed['pool_used']} stores={s_after_seed['stores']}"
        )

        print("\nWARM (different user msgs share the long system prefix):")
        warm = []
        for round_idx in range(args.n_warm):
            for i, u in enumerate(USERS):
                r = stream_ttft(
                    c, args.base, args.model, LONG_SYSTEM, u, args.max_tokens
                )
                warm.append(r)
                print(
                    f"  r{round_idx}.{i}  TTFT={r['ttft_ms']:.0f}ms  total={r['total_ms']:.0f}ms  "
                    f"text={r['text']!r}"
                )

        s_final = get_stats(c, args.base)
        print(f"\nFinal cache stats:\n{json.dumps(s_final, indent=2)}")

    # Summary
    cold_ttft = [r["ttft_ms"] for r in cold]
    warm_ttft = [r["ttft_ms"] for r in warm]
    cold_total = [r["total_ms"] for r in cold]
    warm_total = [r["total_ms"] for r in warm]

    def _stat(xs):
        return f"med={statistics.median(xs):.0f}ms  p90={statistics.quantiles(xs, n=10)[-1]:.0f}ms  min={min(xs):.0f}ms"

    print("\n" + "=" * 60)
    print("TTFT")
    print(f"  cold:  {_stat(cold_ttft)}")
    print(f"  warm:  {_stat(warm_ttft)}")
    saved = statistics.median(cold_ttft) - statistics.median(warm_ttft)
    print(
        f"  saved: {saved:.0f}ms ({saved / statistics.median(cold_ttft) * 100:.0f}% of cold TTFT)"
    )

    print("\nTotal latency (max_tokens=", args.max_tokens, "):", sep="")
    print(f"  cold:  {_stat(cold_total)}")
    print(f"  warm:  {_stat(warm_total)}")


if __name__ == "__main__":
    main()

"""Opt-in APC equivalence probe for Qwen VL models.

This is intentionally not part of the lightweight APC unit suite: it loads a
real Qwen VL checkpoint and compares logits/KV state across three paths:

  1. cold full-prefill shape, matching ``generate_step`` chunking
  2. recomputed cached-prefix shape: prefix first, then short suffix
  3. APC warm path: stored prefix K/V, then the same short suffix

Expected result:

  * split-recompute vs APC warm logits are exact
  * split-prefix K/V vs APC warm K/V are exact
  * cold full-prefill vs APC warm may differ slightly because the final
    attention call has a different query length / reduction order
"""

from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm import models as _models
from mlx_vlm.apc import APCManager, make_warm_kv_cache
from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

SYSTEM = "You are a careful technical writer. Answer in clear prose."
UNIT = (
    " Explain the design tradeoffs in virtual-machine implementation, including "
    "instruction encoding, register allocation, dispatch, exceptions, JIT "
    "compilation, memory models, and runtime profiling."
)


def reset_rope(model) -> None:
    lm = model.language_model
    if hasattr(lm, "_position_ids"):
        lm._position_ids = None
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = None


def prompt_ids(model, processor, reps: int) -> Tuple[mx.array, List[int], str]:
    prompt = apply_chat_template(
        processor,
        model.config,
        prompt=SYSTEM + "\n\n" + UNIT * reps,
        num_images=0,
    )
    ids = processor.tokenizer.encode(prompt)
    bos = getattr(processor.tokenizer, "bos_token_id", None)
    if ids and bos is not None and ids[0] == bos:
        ids = ids[1:]
    return mx.array([ids]), ids, prompt


def make_cache(model):
    return _models.cache.make_prompt_cache(model.language_model, max_kv_size=None)


def embed(model, ids):
    out = model.get_input_embeddings(ids, None, mask=None)
    kwargs = {
        k: v for k, v in out.to_dict().items() if k != "inputs_embeds" and v is not None
    }
    return out.inputs_embeds, kwargs


def eval_cache(prompt_cache) -> None:
    mx.eval([c.state for c in prompt_cache])


def feed_all(model, ids, prompt_cache, chunk_size: int) -> None:
    pos = 0
    while pos < ids.shape[1]:
        n = min(chunk_size, ids.shape[1] - pos)
        sub_ids = ids[:, pos : pos + n]
        sub_embeds, kwargs = embed(model, sub_ids)
        model.language_model(
            sub_ids,
            inputs_embeds=sub_embeds,
            cache=prompt_cache,
            n_to_process=n,
            **kwargs,
        )
        eval_cache(prompt_cache)
        pos += n
        mx.clear_cache()


def full_prefill_logprobs(model, ids, chunk_size: int):
    prompt_cache = make_cache(model)
    inputs_embeds, kwargs = embed(model, ids)
    input_ids = ids

    while inputs_embeds.shape[1] > chunk_size:
        n = min(chunk_size, inputs_embeds.shape[1] - 1)
        model.language_model(
            input_ids[:, :n],
            inputs_embeds=inputs_embeds[:, :n],
            cache=prompt_cache,
            n_to_process=n,
            **kwargs,
        )
        eval_cache(prompt_cache)
        inputs_embeds = inputs_embeds[:, n:]
        input_ids = input_ids[:, n:]
        mx.clear_cache()

    out = model.language_model(
        input_ids,
        inputs_embeds=inputs_embeds,
        cache=prompt_cache,
        **kwargs,
    )
    logprobs = out.logits[:, -1, :] - mx.logsumexp(out.logits[:, -1, :])
    mx.eval(logprobs, [c.state for c in prompt_cache])
    return logprobs, prompt_cache


def tail_logprobs(model, tail_ids, prompt_cache):
    inputs_embeds, kwargs = embed(model, tail_ids)
    out = model.language_model(
        tail_ids,
        inputs_embeds=inputs_embeds,
        cache=prompt_cache,
        **kwargs,
    )
    logprobs = out.logits[:, -1, :] - mx.logsumexp(out.logits[:, -1, :])
    mx.eval(logprobs, [c.state for c in prompt_cache])
    return logprobs, prompt_cache


def store_apc_prefix(ids: List[int], prefix_cache, prefix_len: int, block_size: int):
    manager = APCManager(num_blocks=4096, block_size=block_size)
    layer_keys = []
    layer_values = []
    for c in prefix_cache:
        k, v = c.state
        layer_keys.append(k[..., :prefix_len, :])
        layer_values.append(v[..., :prefix_len, :])

    blocks = manager.store_kv_blocks(
        ids[:prefix_len],
        layer_keys,
        layer_values,
    )
    manager.release(blocks)
    matched, matched_tokens = manager.lookup_prefix(ids)
    warm_cache = make_warm_kv_cache(matched)
    mx.eval([c.state for c in warm_cache])
    return matched_tokens, warm_cache


def max_mean_abs(a, b) -> Tuple[float, float]:
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(diff)
    return float(mx.max(diff).item()), float(mx.mean(diff).item())


def top_token(logprobs) -> Tuple[int, float]:
    idx = int(mx.argmax(logprobs, axis=-1).item())
    return idx, float(logprobs[0, idx].item())


def print_logprob_comparison(name: str, a, b) -> Tuple[float, float, bool]:
    max_abs, mean_abs = max_mean_abs(a, b)
    top_a = top_token(a)
    top_b = top_token(b)
    same_top = top_a[0] == top_b[0]
    print(
        f"{name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"top_a={top_a} top_b={top_b} same_top={same_top}"
    )
    return max_abs, mean_abs, same_top


def run_text(model, processor, prompt: str, manager=None, max_tokens: int = 96):
    parts = []
    tokens = []
    last = None
    for chunk in stream_generate(
        model,
        processor,
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        apc_manager=manager,
    ):
        if chunk.text:
            parts.append(chunk.text)
        if chunk.token is not None and (
            last is None or chunk.generation_tokens > last.generation_tokens
        ):
            tokens.append(int(chunk.token))
        last = chunk
    return "".join(parts).strip(), tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--reps", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--stream-tokens", type=int, default=96)
    args = parser.parse_args()

    print(f"Loading {args.model}")
    model, processor = load(args.model)
    ids, ids_list, prompt = prompt_ids(model, processor, args.reps)
    actual = len(ids_list)
    prefix_len = (actual // args.block_size) * args.block_size
    tail_len = actual - prefix_len
    print(f"actual={actual} prefix_len={prefix_len} tail_len={tail_len}")

    reset_rope(model)
    full_logprobs, _ = full_prefill_logprobs(model, ids, args.chunk_size)

    reset_rope(model)
    split_cache = make_cache(model)
    feed_all(model, ids[:, :prefix_len], split_cache, args.chunk_size)
    split_logprobs, split_cache = tail_logprobs(model, ids[:, prefix_len:], split_cache)

    matched_tokens, warm_cache = store_apc_prefix(
        ids_list,
        split_cache,
        prefix_len,
        args.block_size,
    )
    warm_logprobs, warm_cache = tail_logprobs(model, ids[:, prefix_len:], warm_cache)

    print("\nLogprob comparisons")
    print_logprob_comparison(
        "full/chunked vs split-recompute", full_logprobs, split_logprobs
    )
    exact_max, _, _ = print_logprob_comparison(
        "split-recompute vs APC warm", split_logprobs, warm_logprobs
    )
    print_logprob_comparison("full/chunked vs APC warm", full_logprobs, warm_logprobs)
    print(f"matched_tokens={matched_tokens}")

    key_max = 0.0
    value_max = 0.0
    key_mean = 0.0
    value_mean = 0.0
    for split_layer, warm_layer in zip(split_cache, warm_cache):
        split_k, split_v = split_layer.state
        warm_k, warm_v = warm_layer.state
        k_max, k_mean = max_mean_abs(
            split_k[..., :prefix_len, :],
            warm_k[..., :prefix_len, :],
        )
        v_max, v_mean = max_mean_abs(
            split_v[..., :prefix_len, :],
            warm_v[..., :prefix_len, :],
        )
        key_max = max(key_max, k_max)
        value_max = max(value_max, v_max)
        key_mean += k_mean
        value_mean += v_mean

    print(
        "KV split-prefix vs APC warm: "
        f"key_max={key_max:.6g} value_max={value_max:.6g} "
        f"avg_key_mean={key_mean / len(split_cache):.6g} "
        f"avg_value_mean={value_mean / len(split_cache):.6g}"
    )

    print("\nStream reset regression")
    reset_rope(model)
    cold_text, cold_tokens = run_text(
        model, processor, prompt, None, max_tokens=args.stream_tokens
    )

    manager = APCManager(num_blocks=4096, block_size=args.block_size)
    reset_rope(model)
    for _ in stream_generate(
        model,
        processor,
        prompt,
        max_tokens=1,
        temperature=0.0,
        apc_manager=manager,
    ):
        pass
    before = manager.stats_snapshot()
    reset_rope(model)
    warm_text, warm_tokens = run_text(
        model,
        processor,
        prompt,
        manager,
        max_tokens=args.stream_tokens,
    )
    after = manager.stats_snapshot()
    matched = after.get("matched_tokens", 0) - before.get("matched_tokens", 0)
    common = 0
    for cold_token, warm_token in zip(cold_tokens, warm_tokens):
        if cold_token != warm_token:
            break
        common += 1
    print(
        f"matched={matched} exact_text={cold_text == warm_text} "
        f"exact_tokens={cold_tokens == warm_tokens} "
        f"text_similarity={SequenceMatcher(None, cold_text, warm_text).ratio():.4f} "
        f"common_initial_tokens={common}"
    )

    failed = False
    if exact_max != 0:
        print("FAIL: split-recompute and APC warm logits diverged")
        failed = True
    if key_max != 0 or value_max != 0:
        print("FAIL: APC warm K/V does not exactly match recomputed prefix K/V")
        failed = True
    if matched != prefix_len:
        print("FAIL: stream warm path did not match the full block-aligned prefix")
        failed = True
    if common == 0:
        print("FAIL: stream warm path diverged at the first generated token")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

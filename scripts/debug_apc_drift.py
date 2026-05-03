"""Compare logits between cold prefill vs APC warm-start for the same prompt."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm.apc import APCManager, make_warm_kv_cache
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs


def run_cold(model, processor, ids):
    """Run language model on full input ids with empty cache."""
    cache = make_prompt_cache(model.language_model)
    out = model.language_model(ids, cache=cache)
    return out.logits, cache


def run_warm(model, processor, ids, cache):
    """Run language model on a suffix with a pre-built warm cache."""
    out = model.language_model(ids, cache=cache)
    return out.logits, cache


def main():
    model_path = "mlx-community/SmolVLM-Instruct-bf16"
    print("Loading…")
    model, processor = load(model_path)

    user = (
        "You are a marine biologist. Answer in three short sentences without bullets. "
        "Always include the scientific name of any animal you mention. "
        "Why do octopuses have three hearts? Be concise but precise about the "
        "circulatory anatomy and the role of the systemic versus branchial hearts."
    )
    prompt = apply_chat_template(processor, model.config, prompt=user, num_images=0)
    # Manually re-format to include system to mimic the server path
    inputs = prepare_inputs(processor, prompts=prompt, image_token_index=None)
    ids = inputs["input_ids"]
    print("ids.shape:", ids.shape)

    # Cold full prefill
    cold_logits, cold_cache = run_cold(model, processor, ids)
    mx.eval(cold_logits)
    cold_last = cold_logits[:, -1, :]
    cold_argmax = mx.argmax(cold_last, axis=-1).item()
    print(
        f"COLD: top1 token id={cold_argmax}, top5={mx.argsort(cold_last, axis=-1)[..., -5:].tolist()}"
    )

    # Build APC blocks from cold cache, then warm-start.
    bs = 16
    n_full = ids.shape[1] // bs
    print(f"n_full_blocks={n_full}, ids[1]={ids.shape[1]}")
    if n_full < 2:
        print("Prompt too short to slice into >=2 blocks.")
        return

    mgr = APCManager(num_blocks=64, block_size=bs)
    layer_keys = [c.keys[..., : c.offset, :] for c in cold_cache]
    layer_values = [c.values[..., : c.offset, :] for c in cold_cache]
    blocks = mgr.store_kv_blocks(ids.flatten().tolist(), layer_keys, layer_values)
    print(f"stored {len(blocks)} blocks")

    # Build warm cache from cached blocks (use only first n_full-1 to leave a leftover suffix)
    use = blocks[: n_full - 1]
    warm_cache = make_warm_kv_cache(use)
    prefix_len = (n_full - 1) * bs
    print(f"warm prefix_len={prefix_len}, suffix_len={ids.shape[1] - prefix_len}")

    suffix_ids = ids[:, prefix_len:]
    warm_logits, warm_cache = run_warm(model, processor, suffix_ids, warm_cache)
    mx.eval(warm_logits)
    warm_last = warm_logits[:, -1, :]
    warm_argmax = mx.argmax(warm_last, axis=-1).item()
    print(
        f"WARM: top1 token id={warm_argmax}, top5={mx.argsort(warm_last, axis=-1)[..., -5:].tolist()}"
    )

    # Diff
    diff = mx.abs(cold_last - warm_last)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    print(f"max |Δ| = {max_diff:.6f}, mean |Δ| = {mean_diff:.6f}")

    # Always check K diff and per-position drift — argmax matching just means
    # we got lucky on this prompt.
    print("\nPer-layer K[0:prefix_len] diff (cold cache vs warm cache):")
    for li in range(len(cold_cache)):
        cached_k = cold_cache[li].keys[..., :prefix_len, :]
        warm_layer_k = warm_cache[li].keys[..., :prefix_len, :]
        dk = mx.max(mx.abs(cached_k - warm_layer_k)).item()
        cached_v = cold_cache[li].values[..., :prefix_len, :]
        warm_layer_v = warm_cache[li].values[..., :prefix_len, :]
        dv = mx.max(mx.abs(cached_v - warm_layer_v)).item()
        if dk + dv > 0 or li < 3:
            print(f"  layer {li}: max|ΔK|={dk:.4e}, max|ΔV|={dv:.4e}")

    # Compare suffix logits position by position
    print("\nPer-position logit drift over the suffix:")
    suffix_len = ids.shape[1] - prefix_len
    per_pos_max = []
    for p in range(suffix_len):
        d = mx.max(
            mx.abs(cold_logits[:, prefix_len + p, :] - warm_logits[:, p, :])
        ).item()
        per_pos_max.append(d)
    print(
        f"  max|Δ logits|: min={min(per_pos_max):.4e}, max={max(per_pos_max):.4e}, "
        f"last={per_pos_max[-1]:.4e}"
    )

    # Argmax flips per position
    flips = 0
    for p in range(suffix_len):
        ca = mx.argmax(cold_logits[:, prefix_len + p, :], axis=-1).item()
        wa = mx.argmax(warm_logits[:, p, :], axis=-1).item()
        if ca != wa:
            flips += 1
    print(f"  argmax flips on suffix positions: {flips}/{suffix_len}")


if __name__ == "__main__":
    main()

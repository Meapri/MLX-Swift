"""Direct test of v2 mixed warm/cold prefill via BatchGenerator.insert.

Bypasses the server so all 4 prompts arrive in the same admission window
and the mixed-batch path actually gets B>1 inputs. Compares wall-clock
vs single-prompt sequential prefill.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm import load
from mlx_vlm.apc import APCManager
from mlx_vlm.generate import BatchGenerator
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

SHARED_SYSTEM = (
    "You are a careful technical writer producing reference material. "
    "Always reply with a multi-paragraph explanation: at least three paragraphs, "
    "with concrete examples and historical context where applicable. "
    "Avoid bullet points; use plain prose only. Do not reference earlier turns."
)
USERS = [
    "Explain how a stack-based virtual machine differs from a register-based one.",
    "Describe the role of merkle trees in modern version control systems.",
    "Walk through how DNS resolution works for a typical web request.",
    "Explain reference counting versus tracing garbage collection.",
]


def _get_inputs(model, processor, prompt_text):
    formatted = apply_chat_template(processor, model.config, prompt=prompt_text, num_images=0)
    inputs = prepare_inputs(processor, prompts=formatted, image_token_index=None)
    input_ids = inputs["input_ids"]
    embed = model.get_input_embeddings(input_ids, None)
    gen_kwargs = embed.to_dict()
    return input_ids.squeeze(0).tolist(), gen_kwargs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/SmolVLM-Instruct-bf16")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()

    print(f"Loading {args.model}…")
    model, processor = load(args.model)
    apc = APCManager(num_blocks=2048, block_size=16)

    bg = BatchGenerator(
        model.language_model,
        processor,
        sampler=lambda x: mx.argmax(x, axis=-1),
        max_tokens=args.max_tokens,
        apc_manager=apc,
    )

    # Warm-up: prime the system-prompt blocks via a single request.
    print("Warm-up…")
    pids, pkw = _get_inputs(model, processor, f"{SHARED_SYSTEM}\n\n{USERS[0]}")
    prompts = [pids]
    kwargs = [pkw]
    bg.insert(prompts=prompts, max_tokens=[args.max_tokens], prompt_kwargs=kwargs)
    while bg.has_work:
        bg.next()
    print("APC after warm-up:", apc.stats_snapshot())

    # Now prepare 4 prompts (sharing the system prefix) and admit them all at once.
    prompts = []
    kwargs = []
    for u in USERS:
        ids, kw = _get_inputs(model, processor, f"{SHARED_SYSTEM}\n\n{u}")
        prompts.append(ids)
        kwargs.append(kw)

    apc.reset_stats()
    print(f"Admitting {len(prompts)} prompts in a single batch…")
    t0 = time.perf_counter()
    uids = bg.insert(
        prompts=prompts,
        max_tokens=[args.max_tokens] * len(prompts),
        prompt_kwargs=kwargs,
    )
    while bg.has_work:
        bg.next()
    elapsed = time.perf_counter() - t0
    print(f"4-batch wall-clock: {elapsed:.2f}s")
    print(f"APC stats: {apc.stats_snapshot()}")


if __name__ == "__main__":
    main()

"""Direct-API APC disk bench — bypasses FastAPI/HTTP overhead so the
``prompt_tps`` reported by ``stream_generate`` reflects only the model's
internal prefill timing (well, prefill + cache promotion + first-token).

Compare these numbers to the server bench: any gap is HTTP / scheduler /
SSE overhead. The model-internal ``prompt_tps`` is computed inside
``generate_step`` as ``total_prompt_tokens / prompt_time``, where
``total_prompt_tokens`` includes cached tokens and ``prompt_time`` is
"post-embed → first-token" — so warm rounds still report inflated tps,
just without the HTTP slice.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm import load
from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template


SYSTEM = (
    "You are a careful technical writer producing detailed reference "
    "material. Always reply in full prose paragraphs without bullets."
)

_TEST_PREFIX = (
    "Explain in detail how a stack-based virtual machine differs from a "
    "register-based one. Cover instruction encoding density, decode "
    "complexity, register allocation strategy, exception handling, JIT "
    "considerations, and historical context for JVM, CLR, Lua, and Wasm. "
)

PAD = [
    "Discuss memory model implications.",
    "Compare to dataflow VMs.",
    "Address peephole optimisation differences.",
    "Note implementation in compilers like LLVM.",
    "Mention how the Lua VM evolved across versions.",
    "Describe instruction selection on a register-based target.",
]


def make_test_user(target_tokens: int) -> str:
    out = _TEST_PREFIX
    i = 0
    while len(out.split()) * 3.3 < target_tokens:
        out += PAD[i % len(PAD)] + " "
        i += 1
    return out


def make_filler(idx: int, target_words: int = 160) -> str:
    base = (
        f"unique-tag-{idx:05d} alpha bravo charlie delta echo foxtrot golf "
        f"hotel india juliet kilo lima mike november oscar papa quebec "
    )
    out = base
    while len(out.split()) < target_words:
        out += f" tag{idx:05d}-{len(out)} "
    return out


def run_one(label, model, processor, formatted_prompt, mgr=None):
    t0 = time.perf_counter()
    last = None
    for chunk in stream_generate(
        model, processor, formatted_prompt,
        max_tokens=2, temperature=0.0, apc_manager=mgr,
    ):
        last = chunk
    wall = (time.perf_counter() - t0) * 1000.0
    if last is None:
        print(f"  {label}: no output")
        return None
    print(
        f"  {label}: prompt_tokens={last.prompt_tokens}  "
        f"prompt_tps={last.prompt_tps:.0f}  wall={wall:.0f}ms"
    )
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--test-prompt-tokens", type=int, default=8000)
    ap.add_argument("--fill-prompts", type=int, default=80)
    ap.add_argument("--disk-cap-gb", type=float, default=3.0)
    ap.add_argument("--disk-path", default=None)
    ap.add_argument("--keep-disk", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.model}…")
    t0 = time.perf_counter()
    model, processor = load(args.model)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    user = make_test_user(args.test_prompt_tokens)
    full = f"{SYSTEM}\n\n{user}"
    formatted_prompt = apply_chat_template(
        processor, model.config, prompt=full, num_images=0
    )

    # ---- Cold: no APC at all ----
    print(f"\n[cold] no APC, full prefill of {args.test_prompt_tokens} target tok")
    cold = run_one("cold", model, processor, formatted_prompt, mgr=None)

    disk_root = Path(args.disk_path) if args.disk_path else Path(
        tempfile.mkdtemp(prefix="apc-disk-bench-")
    )
    disk_root.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.disk_cap_gb * (1 << 30)) if args.disk_cap_gb > 0 else None
    print(f"\n[disk] root={disk_root} cap={args.disk_cap_gb:g}GB")

    # ---- Fill phase ----
    disk1 = DiskBlockStore(disk_root, namespace=args.model, max_bytes=max_bytes)
    mgr1 = APCManager(num_blocks=4096, block_size=16, disk=disk1)

    print(f"\n[fill] {args.fill_prompts} unique prompts → exercise APC pool")
    fill_t0 = time.perf_counter()
    for i in range(args.fill_prompts):
        filler_user = make_filler(i)
        filler_full = f"{SYSTEM}\n\n{filler_user}"
        prompt = apply_chat_template(
            processor, model.config, prompt=filler_full, num_images=0
        )
        for _ in stream_generate(
            model, processor, prompt,
            max_tokens=1, temperature=0.0, apc_manager=mgr1,
        ):
            pass
    print(f"  fill done in {time.perf_counter()-fill_t0:.1f}s")

    # Send the test prompt once so its blocks land in the in-memory pool.
    for _ in stream_generate(
        model, processor, formatted_prompt,
        max_tokens=1, temperature=0.0, apc_manager=mgr1,
    ):
        pass

    s_after_fill = mgr1.stats_snapshot()
    print(f"  stats after fill: stores={s_after_fill['stores']} "
          f"mem_evictions={s_after_fill['evictions']} "
          f"pool_used={s_after_fill['pool_used']} "
          f"disk_writes={s_after_fill['disk_writes']} "
          f"disk_files={s_after_fill.get('disk_files', '?')} "
          f"disk_blocks={s_after_fill.get('disk_blocks_indexed', '?')}")
    mgr1.close()

    # ---- Warm-from-disk: new manager, same disk namespace, empty memory pool ----
    disk2 = DiskBlockStore(disk_root, namespace=args.model, max_bytes=max_bytes)
    mgr2 = APCManager(num_blocks=4096, block_size=16, disk=disk2)
    print(f"\n[warm-disk] restarted manager, direct disk prompt-cache restore")
    s_before_disk = mgr2.stats_snapshot()
    warm_disk = run_one("warm-disk", model, processor, formatted_prompt, mgr=mgr2)
    s_after_disk = mgr2.stats_snapshot()
    disk_match = s_after_disk["matched_tokens"] - s_before_disk["matched_tokens"]
    print(f"    stats: matched_tokens={disk_match} "
          f"disk_hits={s_after_disk['disk_hits']} "
          f"pool_used={s_after_disk['pool_used']}")

    print(f"\n[warm-disk-repeat] same manager, immediate repeat")
    s_before_repeat = mgr2.stats_snapshot()
    warm_disk_repeat = run_one("warm-disk-repeat", model, processor, formatted_prompt, mgr=mgr2)
    s_after_repeat = mgr2.stats_snapshot()
    repeat_match = s_after_repeat["matched_tokens"] - s_before_repeat["matched_tokens"]
    print(f"    stats: matched_tokens={repeat_match} "
          f"disk_hits={s_after_repeat['disk_hits']} pool_used={s_after_repeat['pool_used']}")
    mgr2.close()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"Model: {args.model}")
    print(f"Test prompt: ~{args.test_prompt_tokens} target tokens "
          f"(actual={cold.prompt_tokens})")
    print("=" * 60)
    print(f"  cold       prompt_tps = {cold.prompt_tps:.0f}")
    if warm_disk:
        print(f"  warm-disk  prompt_tps = {warm_disk.prompt_tps:.0f}  "
              f"(matched {disk_match} tok from disk)")
    if warm_disk_repeat:
        print(f"  repeat     prompt_tps = {warm_disk_repeat.prompt_tps:.0f}  "
              f"(matched {repeat_match} tok from disk)")

    if args.disk_path is None and not args.keep_disk:
        shutil.rmtree(disk_root, ignore_errors=True)
    else:
        print(f"Disk dir: {disk_root}")


if __name__ == "__main__":
    main()

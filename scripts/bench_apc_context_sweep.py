"""Direct APC context sweep for cold, warm-memory, and warm-disk paths."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_vlm import load  # noqa: E402
from mlx_vlm.apc import APCManager, DiskBlockStore  # noqa: E402
from mlx_vlm.generate import stream_generate  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402


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

SUFFIX = (
    "Now continue from that document with a short implementation note. "
    "Focus on one concrete runtime transition, include the immediate tradeoff, "
    "and keep the continuation consistent with the preceding context. "
)


@dataclass
class RunMetrics:
    context_label: str
    phase: str
    wall_s: float
    ttft_s: float
    prompt_tokens: int
    prompt_tps: float
    gen_tps: float
    peak_gb: float
    matched_tokens: int = 0
    disk_hits: int = 0
    disk_files: int = 0
    disk_bytes: int = 0


def make_test_user(target_tokens: int) -> str:
    out = _TEST_PREFIX
    i = 0
    while len(out.split()) * 3.3 < target_tokens:
        out += PAD[i % len(PAD)] + " "
        i += 1
    return out


def make_suffix(target_tokens: int) -> str:
    out = SUFFIX
    i = 0
    while len(out.split()) * 3.3 < target_tokens:
        out += f" continuation-detail-{i:04d} "
        i += 1
    return out


def prompt_token_count(processor, formatted_prompt: str) -> int:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    encoded = tokenizer(
        formatted_prompt,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return len(input_ids)


def format_user(processor, config, user: str) -> str:
    return apply_chat_template(
        processor,
        config,
        prompt=f"{SYSTEM}\n\n{user}",
        num_images=0,
    )


def build_seed_user_for_eval_context(
    processor,
    config,
    target_eval_tokens: int,
    suffix: str,
) -> tuple[str, int, int]:
    out = _TEST_PREFIX
    pad_unit = " ".join(PAD) + " "

    def eval_count(text: str) -> int:
        return prompt_token_count(processor, format_user(processor, config, text))

    eval_user = f"{out}\n\n{suffix}"
    current = eval_count(eval_user)
    unit_tokens = max(1, eval_count(f"{out}{pad_unit}\n\n{suffix}") - current)
    while current < target_eval_tokens:
        repeat = max(1, (target_eval_tokens - current) // unit_tokens)
        out += pad_unit * repeat
        eval_user = f"{out}\n\n{suffix}"
        current = eval_count(eval_user)

    seed_tokens = eval_count(out)
    return out, seed_tokens, current


def context_label(tokens: int) -> str:
    return f"{tokens // 1000}K" if tokens % 1000 == 0 else str(tokens)


def run_one(
    label: str,
    model,
    processor,
    formatted_prompt: str,
    *,
    apc_manager: Optional[APCManager],
    max_tokens: int,
) -> RunMetrics:
    mx.clear_cache()
    mx.reset_peak_memory()
    start_stats = apc_manager.stats_snapshot() if apc_manager is not None else {}
    t0 = time.perf_counter()
    t_first = None
    last = None
    for chunk in stream_generate(
        model,
        processor,
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        apc_manager=apc_manager,
    ):
        if t_first is None:
            t_first = time.perf_counter()
        last = chunk
    wall_s = time.perf_counter() - t0
    if last is None:
        raise RuntimeError(f"{label} produced no generation chunks")
    end_stats = apc_manager.stats_snapshot() if apc_manager is not None else {}
    return RunMetrics(
        context_label="",
        phase=label,
        wall_s=wall_s,
        ttft_s=(t_first - t0) if t_first is not None else wall_s,
        prompt_tokens=int(last.prompt_tokens),
        prompt_tps=float(last.prompt_tps),
        gen_tps=float(last.generation_tps),
        peak_gb=float(last.peak_memory),
        matched_tokens=int(
            end_stats.get("matched_tokens", 0) - start_stats.get("matched_tokens", 0)
        ),
        disk_hits=int(end_stats.get("disk_hits", 0) - start_stats.get("disk_hits", 0)),
        disk_files=int(end_stats.get("disk_files", 0) or 0),
        disk_bytes=int(end_stats.get("disk_bytes", 0) or 0),
    )


def seed_apc(
    label: str,
    model,
    processor,
    formatted_prompt: str,
    manager: APCManager,
) -> None:
    print(f"    seed {label}...", flush=True)
    for _ in stream_generate(
        model,
        processor,
        formatted_prompt,
        max_tokens=1,
        temperature=0.0,
        apc_manager=manager,
    ):
        pass
    mx.clear_cache()


def fmt_wall(seconds: float) -> str:
    return f"{seconds:.2f}s"


def fmt_rate(rate: float) -> str:
    return f"{rate:,.0f}"


def fmt_peak(gb: float) -> str:
    return f"{gb:.2f}GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[8000, 20000, 50000, 100000],
    )
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--suffix-tokens", type=int, default=256)
    ap.add_argument("--disk-cap-gb", type=float, default=0.0)
    ap.add_argument("--shard-max-blocks", type=int, default=256)
    args = ap.parse_args()

    os.environ["APC_DISK_SHARD_MAX_BLOCKS"] = str(max(1, args.shard_max_blocks))
    disk_cap = int(args.disk_cap_gb * (1 << 30)) if args.disk_cap_gb > 0 else None

    print(f"Loading {args.model}...", flush=True)
    t0 = time.perf_counter()
    model, processor = load(args.model)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Compile/warm basic generation outside measured rows.
    tiny_prompt = apply_chat_template(
        processor, model.config, prompt=f"{SYSTEM}\n\nhello", num_images=0
    )
    print("Warmup...", flush=True)
    for _ in stream_generate(
        model, processor, tiny_prompt, max_tokens=1, temperature=0.0
    ):
        pass
    mx.clear_cache()

    rows: list[RunMetrics] = []
    disk_roots: list[Path] = []

    try:
        for target_tokens in args.contexts:
            label = context_label(target_tokens)
            print(f"\n[{label}] building prompt...", flush=True)
            suffix = make_suffix(args.suffix_tokens)
            user, seed_tokens, eval_tokens = build_seed_user_for_eval_context(
                processor, model.config, target_tokens, suffix
            )
            eval_user = f"{user}\n\n{suffix}"
            seed_prompt = format_user(processor, model.config, user)
            eval_prompt = format_user(processor, model.config, eval_user)
            print(
                f"  tokenized seed={seed_tokens:,} eval={eval_tokens:,}",
                flush=True,
            )

            print(f"  cold...", flush=True)
            cold = run_one(
                "cold",
                model,
                processor,
                eval_prompt,
                apc_manager=None,
                max_tokens=args.max_tokens,
            )
            cold.context_label = label
            rows.append(cold)
            print(
                f"    wall={cold.wall_s:.2f}s ttft={cold.ttft_s:.2f}s "
                f"prompt_tokens={cold.prompt_tokens:,} peak={cold.peak_gb:.2f}GB",
                flush=True,
            )

            n_blocks_est = max(64, cold.prompt_tokens // 16 + 256)
            mem_mgr = APCManager(num_blocks=n_blocks_est, block_size=16)
            try:
                seed_apc("warm-mem", model, processor, seed_prompt, mem_mgr)
                warm_mem = run_one(
                    "warm-mem",
                    model,
                    processor,
                    eval_prompt,
                    apc_manager=mem_mgr,
                    max_tokens=args.max_tokens,
                )
                warm_mem.context_label = label
                rows.append(warm_mem)
                print(
                    f"  warm-mem wall={warm_mem.wall_s:.2f}s "
                    f"ttft={warm_mem.ttft_s:.2f}s "
                    f"matched={warm_mem.matched_tokens:,} "
                    f"peak={warm_mem.peak_gb:.2f}GB",
                    flush=True,
                )
            finally:
                mem_mgr.clear()
                del mem_mgr
                gc.collect()
                mx.clear_cache()

            disk_root = Path(tempfile.mkdtemp(prefix=f"apc-context-{label}-"))
            disk_roots.append(disk_root)
            disk = DiskBlockStore(
                disk_root,
                namespace=args.model,
                max_bytes=disk_cap,
            )
            disk_seed_mgr = APCManager(num_blocks=1, block_size=16, disk=disk)
            try:
                seed_apc("warm-disk", model, processor, seed_prompt, disk_seed_mgr)
                disk._q.join()
                seed_stats = disk_seed_mgr.stats_snapshot()
                print(
                    f"    disk seed files={seed_stats.get('disk_files')} "
                    f"blocks={seed_stats.get('disk_blocks_indexed')} "
                    f"bytes={seed_stats.get('disk_bytes', 0) / (1 << 30):.2f}GB",
                    flush=True,
                )
            finally:
                disk_seed_mgr.close()
                del disk_seed_mgr, disk
                gc.collect()
                mx.clear_cache()

            disk2 = DiskBlockStore(
                disk_root,
                namespace=args.model,
                max_bytes=disk_cap,
            )
            disk_mgr = APCManager(num_blocks=1, block_size=16, disk=disk2)
            try:
                warm_disk = run_one(
                    "warm-disk",
                    model,
                    processor,
                    eval_prompt,
                    apc_manager=disk_mgr,
                    max_tokens=args.max_tokens,
                )
                warm_disk.context_label = label
                rows.append(warm_disk)
                print(
                    f"  warm-disk wall={warm_disk.wall_s:.2f}s "
                    f"ttft={warm_disk.ttft_s:.2f}s "
                    f"matched={warm_disk.matched_tokens:,} "
                    f"disk_hits={warm_disk.disk_hits:,} "
                    f"peak={warm_disk.peak_gb:.2f}GB",
                    flush=True,
                )
            finally:
                disk_mgr.close()
                del disk_mgr, disk2
                gc.collect()
                mx.clear_cache()

        print(
            "\nContext\tPhase\tWall\tSpeedup vs cold\tTTFT\tPrompt tok/s\tGen tok/s\tPeak"
        )
        cold_by_context = {
            row.context_label: row.wall_s for row in rows if row.phase == "cold"
        }
        for row in rows:
            speedup = cold_by_context[row.context_label] / row.wall_s
            print(
                f"{row.context_label}\t{row.phase}\t{fmt_wall(row.wall_s)}\t"
                f"{speedup:.2f}x\t{fmt_wall(row.ttft_s)}\t"
                f"{fmt_rate(row.prompt_tps)}\t{row.gen_tps:.1f}\t"
                f"{fmt_peak(row.peak_gb)}"
            )

        print("\nDetails")
        for row in rows:
            if row.phase == "cold":
                continue
            print(
                f"  {row.context_label} {row.phase}: "
                f"matched={row.matched_tokens:,} "
                f"disk_hits={row.disk_hits:,} disk_files={row.disk_files} "
                f"disk={row.disk_bytes / (1 << 30):.2f}GB"
            )
    finally:
        for root in disk_roots:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

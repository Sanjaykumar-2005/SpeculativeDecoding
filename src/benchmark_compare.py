"""
benchmark_compare.py
====================
THE MAIN EXPERIMENT. Runs all three methods and prints the comparison table:

    1. Qwen 0.5B alone        (draft baseline)
    2. Qwen 1.5B alone        (target baseline -- what we want to speed up)
    3. Speculative decoding   (0.5B drafts, 1.5B verifies)

Then it reports latency, tokens/sec, acceptance rate, accepted/rejected token
counts, and the SPEEDUP of speculative decoding vs. the 1.5B target alone.

Run:
    python -m src.benchmark_compare

Memory strategy for the 4 GB RTX 3050: the two single-model baselines load one
model at a time and free it before the next. The speculative run then loads BOTH
(draft fp16 + target 4-bit) together. See src/config.py for the precision knobs.
"""

from __future__ import annotations

import gc
import sys

import torch
from tabulate import tabulate

# Force UTF-8 stdout: model answers may echo non-cp1252 chars (e.g. "₹") which
# would crash print() on the default Windows console. See benchmark_single.py.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import config
from .benchmark_single import benchmark_model, free
from .prompts import OCR_TASKS, build_chat_prompt
from .speculative_decoding import speculative_generate
from .utils import GenResult, load_model, load_tokenizer, set_seed


def benchmark_speculative(draft_model, target_model, tokenizer) -> GenResult:
    """Average speculative decoding over all OCR tasks (with one warmup)."""
    print("\n=== Benchmarking: Speculative Decoding (0.5B draft + 1.5B target) ===")

    # Warmup (untimed).
    warm = build_chat_prompt(tokenizer, OCR_TASKS[0])
    speculative_generate(draft_model, target_model, tokenizer, warm)

    agg = GenResult()
    n_runs = 0
    for task in OCR_TASKS:
        prompt = build_chat_prompt(tokenizer, task)
        for _ in range(config.N_REPEATS):
            r = speculative_generate(draft_model, target_model, tokenizer, prompt)
            agg.n_tokens += r.n_tokens
            agg.latency_s += r.latency_s
            agg.n_proposed += r.n_proposed
            agg.n_accepted += r.n_accepted
            agg.n_rejected += r.n_rejected
            agg.peak_mem_mb = max(agg.peak_mem_mb, r.peak_mem_mb)
            n_runs += 1
        print(f"  [{task[:28]:28}] -> {r.text.strip()[:60]!r}")

    avg = GenResult(
        n_tokens=agg.n_tokens / n_runs,
        latency_s=agg.latency_s / n_runs,
        peak_mem_mb=agg.peak_mem_mb,
        n_proposed=agg.n_proposed,
        n_accepted=agg.n_accepted,
        n_rejected=agg.n_rejected,
    )
    avg.tokens_per_s = avg.n_tokens / avg.latency_s if avg.latency_s > 0 else 0.0
    avg.acceptance_rate = (agg.n_accepted / agg.n_proposed) if agg.n_proposed else 0.0

    print(f"  -> avg latency      : {avg.latency_s*1000:8.1f} ms")
    print(f"  -> avg tokens/sec   : {avg.tokens_per_s:8.1f}")
    print(f"  -> acceptance rate  : {avg.acceptance_rate*100:8.1f} %")
    print(f"  -> accepted tokens  : {avg.n_accepted}")
    print(f"  -> rejected tokens  : {avg.n_rejected}")
    print(f"  -> peak GPU memory  : {avg.peak_mem_mb:8.1f} MB")
    return avg


def print_table(draft: GenResult, target: GenResult, spec: GenResult):
    """Print the final markdown comparison table requested in the spec."""

    def speedup_vs_target(r: GenResult) -> str:
        if r.latency_s <= 0:
            return "-"
        return f"{target.latency_s / r.latency_s:.2f}x"

    rows = [
        [
            "Qwen 0.5B alone",
            f"{draft.latency_s*1000:.1f} ms",
            f"{draft.tokens_per_s:.1f}",
            "-",
            speedup_vs_target(draft),
        ],
        [
            "Qwen 1.5B alone",
            f"{target.latency_s*1000:.1f} ms",
            f"{target.tokens_per_s:.1f}",
            "-",
            "1.00x (baseline)",
        ],
        [
            "Speculative (0.5B+1.5B)",
            f"{spec.latency_s*1000:.1f} ms",
            f"{spec.tokens_per_s:.1f}",
            f"{spec.acceptance_rate*100:.1f} %",
            speedup_vs_target(spec),
        ],
    ]
    headers = ["Method", "Latency", "Tokens/sec", "Acceptance Rate", "Speedup"]
    print("\n" + "=" * 72)
    print("FINAL COMPARISON  (averaged over all OCR prompts, greedy decoding)")
    print("=" * 72)
    print(tabulate(rows, headers=headers, tablefmt="github"))

    print("\nSpeculative decoding detail:")
    print(f"  accepted tokens : {spec.n_accepted}")
    print(f"  rejected tokens : {spec.n_rejected}")
    print(f"  proposed tokens : {spec.n_proposed}")
    print(f"  acceptance rate : {spec.acceptance_rate*100:.1f}%")
    sp = (target.latency_s / spec.latency_s) if spec.latency_s > 0 else 0.0
    print(f"  speedup vs 1.5B : {sp:.2f}x")
    print("=" * 72)


def main():
    set_seed()
    tokenizer = load_tokenizer(config.TARGET_MODEL_ID)

    # IMPORTANT (Windows + bitsandbytes): a 4-bit quantized model must be loaded
    # only ONCE per process. Re-initializing a second bitsandbytes 4-bit model
    # after freeing the first causes a hard CUDA access violation (segfault) on
    # Windows. So the order below loads the (4-bit) TARGET exactly once and keeps
    # it resident for the speculative phase. The fp16 DRAFT is tiny and safe to
    # load more than once.

    # --- 1) Draft (0.5B) alone -- measured with clean VRAM ---------------
    draft_model = load_model(config.DRAFT_MODEL_ID, config.DRAFT_4BIT)
    draft_res = benchmark_model(draft_model, tokenizer,
                                f"Qwen 0.5B draft ({config.DRAFT_MODEL_ID})")
    free(draft_model)  # fp16 free/reload is safe

    # --- 2) Target (1.5B) alone -- measured with clean VRAM, loaded ONCE -
    target_model = load_model(config.TARGET_MODEL_ID, config.TARGET_4BIT)
    target_res = benchmark_model(target_model, tokenizer,
                                 f"Qwen 1.5B target ({config.TARGET_MODEL_ID})")
    # NOTE: do NOT free the target here -- we reuse this exact instance below.

    # --- 3) Speculative decoding: reload the fp16 draft next to the
    #        already-resident target so BOTH are in VRAM together. ---------
    draft_model = load_model(config.DRAFT_MODEL_ID, config.DRAFT_4BIT)
    spec_res = benchmark_speculative(draft_model, target_model, tokenizer)
    free(draft_model)
    free(target_model)

    # --- Results ----------------------------------------------------------
    print_table(draft_res, target_res, spec_res)


if __name__ == "__main__":
    main()

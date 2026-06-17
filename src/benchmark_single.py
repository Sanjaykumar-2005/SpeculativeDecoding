"""
benchmark_single.py
===================
Benchmark ONE model on its own across all OCR prompts.

Run directly to test a single model:
    python -m src.benchmark_single --model draft
    python -m src.benchmark_single --model target

It measures, per prompt and averaged: latency, tokens generated, tokens/sec,
and peak GPU memory. These are the baselines the speculative run is compared to.
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch

# Model answers can contain non-cp1252 characters (e.g. the Rupee sign "₹" the
# model may echo from an invoice). The default Windows console is cp1252 and
# would raise UnicodeEncodeError on print(); force UTF-8 so output never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from . import config
from .prompts import OCR_TASKS, build_chat_prompt
from .utils import GenResult, greedy_generate, load_model, load_tokenizer, set_seed


def benchmark_model(model, tokenizer, label: str) -> GenResult:
    """Average greedy generation over all OCR tasks (with one warmup)."""
    print(f"\n=== Benchmarking: {label} ===")

    # Warmup (untimed): triggers CUDA kernel autotuning so timings are stable.
    warm_prompt = build_chat_prompt(tokenizer, OCR_TASKS[0])
    greedy_generate(model, tokenizer, warm_prompt)

    totals = GenResult()
    n_runs = 0
    for task in OCR_TASKS:
        prompt = build_chat_prompt(tokenizer, task)
        # repeat each prompt N times and average to reduce noise
        for _ in range(config.N_REPEATS):
            r = greedy_generate(model, tokenizer, prompt)
            totals.n_tokens += r.n_tokens
            totals.latency_s += r.latency_s
            totals.peak_mem_mb = max(totals.peak_mem_mb, r.peak_mem_mb)
            n_runs += 1
        # Show the model's answer once per task so we can sanity-check quality.
        print(f"  [{task[:28]:28}] -> {r.text.strip()[:60]!r}")

    avg = GenResult(
        n_tokens=totals.n_tokens / n_runs,
        latency_s=totals.latency_s / n_runs,
        peak_mem_mb=totals.peak_mem_mb,
    )
    avg.tokens_per_s = avg.n_tokens / avg.latency_s if avg.latency_s > 0 else 0.0

    print(f"  -> avg latency      : {avg.latency_s*1000:8.1f} ms")
    print(f"  -> avg new tokens   : {avg.n_tokens:8.1f}")
    print(f"  -> avg tokens/sec   : {avg.tokens_per_s:8.1f}")
    print(f"  -> peak GPU memory  : {avg.peak_mem_mb:8.1f} MB")
    return avg


def free(model):
    """Release a model from VRAM so the next one fits on the 4 GB card."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["draft", "target"], default="draft")
    args = parser.parse_args()

    set_seed()
    if args.model == "draft":
        model_id, four_bit = config.DRAFT_MODEL_ID, config.DRAFT_4BIT
    else:
        model_id, four_bit = config.TARGET_MODEL_ID, config.TARGET_4BIT

    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, four_bit)
    benchmark_model(model, tokenizer, f"{args.model} ({model_id})")
    free(model)


if __name__ == "__main__":
    main()

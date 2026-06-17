"""
config.py
=========
Central configuration for the speculative decoding POC.

Everything you might want to tweak (model names, precision, how many tokens the
draft proposes per round, generation length) lives here so the rest of the code
stays clean. Read the inline comments — they explain *why* each default was
chosen for a 4 GB RTX 3050 laptop.
"""

from __future__ import annotations

import os

import torch

# ---------------------------------------------------------------------------
# 1. MODELS
# ---------------------------------------------------------------------------
# Speculative decoding REQUIRES the draft and target models to share the SAME
# tokenizer / vocabulary, otherwise a draft token id has a different meaning to
# the target and verification is nonsense. The Qwen2.5 family all share one
# tokenizer (vocab size 151936), so 0.5B + 1.5B is a valid pair — and so is
# your future plan of 3B (draft) + 32B (target), which also share this vocab.
# These can be overridden with environment variables, which is handy if you have
# pre-downloaded the weights to a local folder (e.g. on a flaky connection) or
# want to swap in the 3B/32B pair later without editing code:
#   set DRAFT_MODEL_ID=...   /   set TARGET_MODEL_ID=...   (or a local path)
DRAFT_MODEL_ID = os.environ.get(
    "DRAFT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"   # small, fast "guesser"
)
TARGET_MODEL_ID = os.environ.get(
    "TARGET_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"  # bigger, accurate "verifier"
)

# ---------------------------------------------------------------------------
# 2. DEVICE & PRECISION
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# float16 is the right compute dtype for the RTX 3050 (no bf16 tensor cores on
# Ampere consumer cards for this size; fp16 is well supported).
DTYPE = torch.float16

# --- The critical memory knob for 4 GB VRAM -------------------------------
# Weights in fp16:  0.5B ~= 1.0 GB,  1.5B ~= 3.1 GB  -> together ~4.1 GB.
# A 4 GB card (shared with the Windows desktop) cannot hold both in fp16.
# So we load the TARGET in 4-bit (NF4) which shrinks it to ~0.9 GB. The draft
# stays in fp16 (it is tiny). Combined footprint then ~2 GB + KV cache -> fits.
#
# We keep the target precision IDENTICAL between the "target alone" benchmark
# and the speculative run, so the measured speedup is apples-to-apples.
#
# If you move this code to a big GPU (e.g. H200), set TARGET_4BIT = False to run
# everything in fp16/bf16 — the algorithm is unchanged.
TARGET_4BIT = True
DRAFT_4BIT = False   # draft is small enough to keep in fp16 for max speed

# ---------------------------------------------------------------------------
# 3. SPECULATIVE DECODING PARAMETERS
# ---------------------------------------------------------------------------
# GAMMA = how many tokens the draft model proposes per verification round.
#   - Too small -> you barely amortize the target's forward pass.
#   - Too large -> the draft drifts from the target, acceptance drops, and you
#                  waste compute on tokens that get rejected.
# 4 is a good starting point for a 0.5B/1.5B pair. On this 4 GB laptop a sweep
# (see README) found gamma=6 best: 100% acceptance and the lowest speculative
# latency. Tune it and watch the acceptance rate vs. speedup.
GAMMA = 6

# How many new tokens to generate per prompt during benchmarking.
MAX_NEW_TOKENS = 64

# Greedy decoding (do_sample=False everywhere) makes the experiment
# DETERMINISTIC and makes greedy speculative decoding provably LOSSLESS:
# the speculative output is token-for-token identical to running the target
# alone. That is exactly what you want for an OCR field-extraction task where
# you care about correctness, not creativity.
DO_SAMPLE = False

# Number of timed repetitions per prompt (results are averaged). The first,
# untimed run is a warmup so CUDA kernel compilation / autotuning doesn't
# pollute the measurement.
N_REPEATS = 3

# ---------------------------------------------------------------------------
# 4. MISC
# ---------------------------------------------------------------------------
SEED = 1234
RESULTS_DIR = "results"

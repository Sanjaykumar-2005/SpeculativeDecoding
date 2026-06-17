"""
utils.py
========
Shared helpers: model loading, memory/timing measurement, and a plain greedy
generation loop used for the single-model baselines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import config


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class GenResult:
    """Everything we measure for one generation run."""
    text: str = ""                 # the decoded answer
    n_tokens: int = 0              # number of NEW tokens generated
    latency_s: float = 0.0         # wall-clock seconds for generation
    tokens_per_s: float = 0.0      # n_tokens / latency_s
    peak_mem_mb: float = 0.0       # peak CUDA memory during the run
    # Speculative-only stats (left at defaults for single-model runs):
    n_proposed: int = 0            # total draft tokens proposed
    n_accepted: int = 0            # draft tokens accepted by the target
    n_rejected: int = 0            # draft tokens rejected by the target
    acceptance_rate: float = 0.0   # n_accepted / n_proposed
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------
def load_tokenizer(model_id: str):
    """Both models share a tokenizer; we load from the target by convention."""
    tok = AutoTokenizer.from_pretrained(model_id)
    # Qwen has a pad token, but be defensive for batching/eos handling.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id: str, four_bit: bool):
    """
    Load a causal-LM, optionally quantized to 4-bit (NF4) so it fits in 4 GB.

    4-bit uses bitsandbytes: weights are stored as NF4 and dequantized on the
    fly during matmul. The KV cache and activations stay in fp16, so generation
    quality is close to fp16 while the weight footprint drops ~4x.
    """
    common = dict(torch_dtype=config.DTYPE, low_cpu_mem_usage=True)

    if four_bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",             # NF4 = best quality 4-bit
            bnb_4bit_compute_dtype=config.DTYPE,   # math runs in fp16
            bnb_4bit_use_double_quant=True,        # extra ~0.4 bit/param saving
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0}, **common
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
        model.to(config.DEVICE)

    model.eval()  # disable dropout etc.; we never train here
    return model


# ---------------------------------------------------------------------------
# Memory / timing helpers
# ---------------------------------------------------------------------------
def reset_peak_memory():
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if config.DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def cuda_sync():
    """CUDA kernels are async; sync before reading the clock or timings lie."""
    if config.DEVICE == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Plain greedy generation (single-model baseline)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = None) -> GenResult:
    """
    Standard autoregressive greedy decoding using HF's `generate`. This is the
    BASELINE each model is measured against. `generate` internally uses a KV
    cache, so this is an efficient, fair baseline (one target forward per token).

    max_new_tokens defaults to config.MAX_NEW_TOKENS; pass a larger value for
    long structured outputs (e.g. full-document JSON extraction).
    """
    mnt = max_new_tokens if max_new_tokens is not None else config.MAX_NEW_TOKENS
    inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    reset_peak_memory()
    cuda_sync()
    t0 = time.perf_counter()

    # IMPORTANT for losslessness: Qwen's generation_config ships do_sample=True,
    # temperature/top_p/top_k AND repetition_penalty=1.1. `generate` would apply
    # the repetition penalty, but our hand-rolled speculative loop uses RAW argmax
    # with no penalty. To keep the baseline token-identical to speculative (so the
    # "lossless" claim actually holds), we force PURE greedy here: penalty off,
    # sampling knobs unset, and the same single EOS the speculative loop stops on.
    out = model.generate(
        **inputs,
        max_new_tokens=mnt,
        do_sample=False,                    # greedy (argmax)
        num_beams=1,
        temperature=None,                   # unset -> silences warnings + ensures greedy
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,             # KEY: no penalty, match speculative
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    cuda_sync()
    latency = time.perf_counter() - t0

    new_tokens = out[0, prompt_len:]
    n_tokens = int(new_tokens.shape[0])
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return GenResult(
        text=text,
        n_tokens=n_tokens,
        latency_s=latency,
        tokens_per_s=(n_tokens / latency) if latency > 0 else 0.0,
        peak_mem_mb=peak_memory_mb(),
    )


def set_seed(seed: int = config.SEED):
    torch.manual_seed(seed)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

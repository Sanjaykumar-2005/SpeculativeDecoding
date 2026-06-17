"""
speculative_decoding.py
=======================
The core algorithm. This module implements GREEDY speculative decoding, which is
provably LOSSLESS: the tokens it emits are identical to what the target model
would produce on its own with greedy decoding — but (ideally) faster.

------------------------------------------------------------------------------
WHY SPECULATIVE DECODING IS FASTER
------------------------------------------------------------------------------
Generating one token from a big model normally needs one full forward pass of
that big model. The forward pass is dominated by reading the model's weights
from GPU memory (it is "memory-bandwidth bound"), and it costs almost the same
whether you push 1 token or 5 tokens through it at once.

Idea: let a CHEAP draft model guess the next few tokens, then run the EXPENSIVE
target model ONCE over all those guesses to check them in parallel. Every guess
the target agrees with is a token we got "for free" — we verified several tokens
with a single big-model forward pass instead of one pass per token.

------------------------------------------------------------------------------
THE ALGORITHM, ONE ROUND AT A TIME  (gamma = number of guesses per round)
------------------------------------------------------------------------------
1. PROPOSE (draft): the draft model autoregressively generates `gamma` tokens,
   one at a time (cheap, because the draft is small).

2. VERIFY (target): the target model does ONE forward pass over those `gamma`
   draft tokens. Because of the causal mask, this single pass yields the
   target's own "what token would I pick here?" prediction at every one of the
   `gamma` positions — in parallel.

3. COMPARE: walk left-to-right. While draft token i == target's prediction at
   position i, ACCEPT it. At the first mismatch, REJECT that token and stop.

4. CORRECT / BONUS:
   - If a token was rejected, replace it with the TARGET's own token at that
     position (the "correction"). Everything after it is thrown away.
   - If ALL `gamma` tokens were accepted, the same target forward pass already
     gave us one extra "bonus" token (its prediction for the position right
     after the last guess) — a free token.
   Either way, each round commits at least 1 and at most gamma+1 tokens, and
   every committed token is exactly what the target alone would have produced.

5. REPEAT until we hit EOS or the token budget.

ACCEPTANCE RATE = total accepted draft tokens / total proposed draft tokens.
Higher acceptance => the draft is a good mimic of the target => bigger speedup.

------------------------------------------------------------------------------
KV-CACHE BOOKKEEPING (the tricky part)
------------------------------------------------------------------------------
Both models keep a KV cache so we never re-process old tokens. When the target
rejects some guesses, those rejected tokens were already written into BOTH
caches during propose/verify, so we must CROP both caches back to the length of
the accepted prefix. The single "correction"/"bonus" token is intentionally
left OUT of the caches and is fed in at the start of the next round, which keeps
the loop uniform and the off-by-one bookkeeping correct.
"""

from __future__ import annotations

import time

import torch
from transformers import DynamicCache

from . import config
from .utils import GenResult, cuda_sync, peak_memory_mb, reset_peak_memory


@torch.no_grad()
def speculative_generate(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    gamma: int = config.GAMMA,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    verbose: bool = False,
) -> GenResult:
    device = config.DEVICE
    eos_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]            # [1, prompt_len]
    prompt_len = input_ids.shape[1]

    # Counters for the metrics we report.
    n_proposed = n_accepted = n_rejected = 0
    generated_ids = []                          # NEW tokens only (the answer)

    reset_peak_memory()
    cuda_sync()
    t0 = time.perf_counter()

    # --- Caches: one per model. DynamicCache grows as we feed tokens and
    #     supports .crop() to roll back rejected tokens. -------------------
    draft_cache = DynamicCache()
    target_cache = DynamicCache()

    # --- PREFILL ---------------------------------------------------------
    # Feed prompt[:-1] into both models so their caches hold the whole prompt
    # except its last token. `last_token` is then ingested at the top of the
    # main loop, which keeps the loop body uniform (it always starts by
    # ingesting exactly one pending token).
    if prompt_len > 1:
        draft_model(input_ids[:, :-1], past_key_values=draft_cache, use_cache=True)
        target_model(input_ids[:, :-1], past_key_values=target_cache, use_cache=True)
    last_token = input_ids[:, -1:]              # [1, 1] — the pending token

    while len(generated_ids) < max_new_tokens:
        # =================================================================
        # STEP A — ingest the single pending token into BOTH models.
        #   This updates both caches to the same committed length and gives us
        #   each model's prediction for the FIRST upcoming position.
        # =================================================================
        d_out = draft_model(last_token, past_key_values=draft_cache, use_cache=True)
        t_out = target_model(last_token, past_key_values=target_cache, use_cache=True)
        draft_next_logits = d_out.logits[:, -1, :]    # draft's pred for draft-pos 0
        target_logits_0 = t_out.logits[:, -1, :]      # target's pred for draft-pos 0

        # Length of both caches now = number of committed tokens. We crop back
        # to this after deciding how many guesses to keep.
        base_len = draft_cache.get_seq_length()

        # =================================================================
        # STEP B — PROPOSE: draft generates `gamma` tokens autoregressively.
        # =================================================================
        draft_tokens = []
        logits = draft_next_logits
        for _ in range(gamma):
            tok = logits.argmax(dim=-1, keepdim=True)          # greedy [1,1]
            draft_tokens.append(tok)
            d_out = draft_model(tok, past_key_values=draft_cache, use_cache=True)
            logits = d_out.logits[:, -1, :]
        draft_seq = torch.cat(draft_tokens, dim=1)             # [1, gamma]
        n_proposed += gamma
        # draft_cache now holds: committed + gamma draft tokens.

        # =================================================================
        # STEP C — VERIFY: ONE target forward over all gamma draft tokens.
        #   Feeding the gamma tokens yields, at each position, the target's
        #   prediction for the NEXT token. Combined with target_logits_0
        #   (its prediction for position 0, from Step A), we get the target's
        #   own choice at every one of the gamma draft positions, plus one
        #   "bonus" prediction for the position after the last guess.
        # =================================================================
        t_out = target_model(draft_seq, past_key_values=target_cache, use_cache=True)
        # predictions for draft positions 0..gamma-1:
        target_step_logits = torch.cat(
            [target_logits_0.unsqueeze(1), t_out.logits[:, :-1, :]], dim=1
        )                                                       # [1, gamma, V]
        bonus_logits = t_out.logits[:, -1, :]                  # pred AFTER last guess
        target_preds = target_step_logits.argmax(dim=-1)       # [1, gamma]
        # target_cache now holds: committed + gamma draft tokens.

        # =================================================================
        # STEP D — COMPARE left-to-right; accept the matching prefix.
        # =================================================================
        accepted = 0
        for i in range(gamma):
            if draft_seq[0, i].item() == target_preds[0, i].item():
                accepted += 1
            else:
                break                       # first disagreement ends the round
        n_accepted += accepted
        n_rejected += (gamma - accepted)

        # =================================================================
        # STEP E — pick the correction / bonus token.
        #   - partial accept: use the target's own token at the mismatch.
        #   - full accept   : use the free bonus token from the same pass.
        # =================================================================
        if accepted == gamma:
            correction = bonus_logits.argmax(dim=-1, keepdim=True)     # free token
        else:
            correction = target_preds[:, accepted:accepted + 1]        # target's fix

        # =================================================================
        # STEP F — roll back BOTH caches to the accepted prefix.
        #   The rejected draft tokens are still sitting in the caches from
        #   steps B and C; crop them away. The correction token is NOT added
        #   to the caches here — it becomes next round's pending `last_token`.
        # =================================================================
        keep_len = base_len + accepted
        draft_cache.crop(keep_len)
        target_cache.crop(keep_len)

        # =================================================================
        # STEP G — commit tokens, handle EOS, set up next round.
        # =================================================================
        committed = draft_seq[0, :accepted].tolist() + [correction.item()]
        for tid in committed:
            if len(generated_ids) >= max_new_tokens:
                break
            generated_ids.append(tid)
            if tid == eos_id:
                break

        last_token = correction             # ingested at top of next loop

        if verbose:
            print(f"  round: proposed={gamma} accepted={accepted} "
                  f"-> committed {len(committed)} tokens "
                  f"(total {len(generated_ids)})")

        if eos_id is not None and eos_id in committed:
            break

    cuda_sync()
    latency = time.perf_counter() - t0

    # Trim any overshoot past the budget, then decode.
    generated_ids = generated_ids[:max_new_tokens]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)

    return GenResult(
        text=text,
        n_tokens=n_tokens,
        latency_s=latency,
        tokens_per_s=(n_tokens / latency) if latency > 0 else 0.0,
        peak_mem_mb=peak_memory_mb(),
        n_proposed=n_proposed,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        acceptance_rate=(n_accepted / n_proposed) if n_proposed > 0 else 0.0,
        extra={"gamma": gamma},
    )

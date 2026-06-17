# Speculative Decoding — How It Works, Step by Step

This document explains the technique implemented in
`src/speculative_decoding.py`. It is written to build intuition first, then map
that intuition onto the exact code.

---

## 1. The problem: autoregressive generation is serial and memory-bound

A normal LLM generates text one token at a time. To produce token *t* it runs a
full forward pass of the whole model, then feeds that token back to produce
token *t+1*, and so on. Two facts make this slow:

1. **It's serial.** Token *t+1* literally cannot start until token *t* exists.
2. **Each step is memory-bandwidth bound.** The forward pass spends most of its
   time *reading the model weights from GPU memory*, not doing math. Crucially,
   reading the weights to process **1** token costs almost the same as reading
   them to process **5** tokens at once. The GPU is idle-ish either way.

So a big model generating 64 tokens does 64 expensive, mostly-serial passes.

---

## 2. The idea: guess cheaply, verify in bulk

What if a small, cheap model **guesses** the next few tokens, and the big model
**checks all the guesses in a single forward pass**?

- The small (draft) model is fast, so guessing is cheap.
- Checking *k* guesses costs the big (target) model basically **one** forward
  pass — the same pass would otherwise produce just one token.

Every guess the target agrees with is a token we obtained without paying for a
separate big-model step. If the draft is a decent mimic of the target, most
guesses are right, and we generate several tokens per target pass instead of one.

The catch that makes it *safe*: we never blindly trust the draft. The target
verifies, and the final output is **exactly** what the target would have
produced on its own (under greedy decoding). Speculative decoding changes the
*speed*, not the *result*. This is why it's called **lossless**.

---

## 3. One round in detail

Let `gamma` = number of guesses per round (default 4 in `config.py`).

### Step 1 — The draft model PROPOSES `gamma` tokens
The draft generates `gamma` tokens autoregressively, one at a time, just like
normal generation — but it's the *small* model, so this is fast.

```
prompt: "... TOTAL AMOUNT "
draft guesses (gamma=4):  ["15", ",", "871", ".00"]
```
> Code: **Step B** in `speculative_generate`. Each guess is `argmax` of the
> draft's logits; we keep a KV cache so each step only processes one new token.

### Step 2 — The target model VERIFIES all guesses in ONE pass
We feed the `gamma` draft tokens to the target in a **single** forward pass.
Thanks to the causal attention mask, that one pass produces, at every position,
the target's own answer to *"what token would I put here?"* — for all `gamma`
positions in parallel, plus one extra prediction for the position right after
the last guess (the **bonus** slot).

```
position:        0      1      2       3      (bonus)
draft guess:    "15"   ","   "871"  ".00"
target's pick:  "15"   ","   "871"  ".00"   "<eos>"
```
> Code: **Step C**. The target's pick at position 0 comes from Step A's
> `target_logits_0`; picks for positions 1..gamma-1 come from this pass;
> `bonus_logits` is the prediction for the slot after the last guess.

### Step 3 — COMPARE left to right; accept the matching prefix
Walk through the positions. While `draft_guess[i] == target_pick[i]`, **accept**.
Stop at the first mismatch.

- **All match** (as above): accept all 4, *and* take the free bonus token →
  5 tokens committed from one target pass.
- **Mismatch at position i**: accept positions `0..i-1`, **reject** position `i`
  and everything after it.

> Code: **Step D** (the accept loop) and **Step E** (choosing the replacement).

### Step 4 — How REJECTED tokens are regenerated
When the target rejects the draft's guess at position `i`, we **don't** ask the
draft again. We already know the right token: it's the **target's own pick** at
position `i`, which we computed in Step 2 (the "correction"). We commit that and
discard the rest of the draft's guesses for this round.

```
draft guessed:  ["INV", "-", "2026", "-9999"]   (last one wrong)
target picks :  ["INV", "-", "2026", "-0048" ...]
accept "INV","-","2026"  (3 tokens), reject "-9999",
commit target's correction "-0048", start a new round.
```
This guarantees progress: **every round commits at least 1 token** (the
correction) even in the worst case where the draft is wrong immediately.

> Code: **Step E** picks `correction`; **Step G** commits accepted prefix + correction.

### Step 5 — How accepted tokens are REUSED (KV cache bookkeeping)
Both models keep a KV cache so past tokens are never recomputed. During Steps 1
and 2, the rejected guesses got written into both caches. So after deciding how
many to keep, we **crop both caches** back to `committed_length + accepted`. The
correction token is deliberately left *out* of the caches and fed in at the top
of the next round — this keeps the loop uniform and the indexing correct.

> Code: **Step F** (`draft_cache.crop(...)`, `target_cache.crop(...)`) and
> **Step A** (ingesting the single pending token at the start of each round).

### Step 6 — REPEAT
Continue until an end-of-sequence token is committed or the token budget is hit.

---

## 4. How the acceptance rate is calculated

```
acceptance_rate = total_accepted_draft_tokens / total_proposed_draft_tokens
```

- `total_proposed` increases by `gamma` every round (Step B: `n_proposed += gamma`).
- `total_accepted` increases by the length of the matching prefix each round
  (Step D: `n_accepted += accepted`).
- `total_rejected = total_proposed - total_accepted`.

The **bonus** token (from a fully-accepted round) is *not* counted as a proposed
or accepted draft token — it was produced by the target, not guessed by the
draft. Counting only draft guesses keeps the acceptance rate a clean measure of
"how good a mimic the draft is."

**Interpretation**
- High acceptance (e.g. 70–90%) → the draft tracks the target well → big speedup.
- Low acceptance → the target keeps overriding guesses → little/negative speedup
  (you paid for the draft work for nothing). Lower `GAMMA` or use a better draft.

OCR field extraction tends to have **high** acceptance: the output is short,
templated, and copied largely verbatim from the document, so even a 0.5B model
guesses most tokens correctly.

---

## 5. How the speedup is calculated

```
speedup = latency(target_alone) / latency(speculative)
```

Both are measured on the **same prompts**, the **same target weights/precision**,
and **greedy** decoding (so their *outputs* are identical and only speed differs).
`speculative_generate` times the whole loop with `time.perf_counter()`, calling
`torch.cuda.synchronize()` before reading the clock because CUDA kernels run
asynchronously and would otherwise report misleadingly small times.

### A losslessness gotcha: logit-processing must match on BOTH paths
"Lossless" only holds if the target-alone baseline and the speculative loop turn
logits into tokens the *same* way. Qwen's `generation_config.json` ships
`do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, and
**`repetition_penalty=1.1`**. The baseline uses `model.generate`, which *applies*
the repetition penalty; our hand-rolled speculative loop uses **raw argmax** with
no penalty. So the two paths were optimizing different distributions — on the
long production prompt they diverged (target emitted `₹ 15,871.00`, speculative
`15871.00`) and acceptance fell. The fix: force the baseline to **pure greedy**
(`do_sample=False`, `repetition_penalty=1.0`, sampling knobs unset, same single
EOS). Lesson for the H200: whatever logit processing you run in production
(penalties, sampling) must be applied **identically** in the draft proposal, the
target verification, and the acceptance test — otherwise speculative decoding is
no longer lossless.

### Why the laptop speedup is actually sub-1x (measured: ~0.86–0.91x)
Measured honestly on this 4 GB RTX 3050 (warmed, interleaved), speculative
decoding is *slightly slower* than the 1.5B alone — **even at 100% acceptance**.
This is the expected result here, not a bug:

- The target runs in **4-bit** on a tiny GPU, so its per-token forward is already
  cheap/memory-bound. Speculative decoding amortizes the target's forward pass,
  but that pass simply isn't expensive enough here for the amortization to pay off.
- The draft (0.5B fp16) costs a real fraction of the target's time, and each round
  adds `gamma` sequential draft steps plus KV-cache crop/bookkeeping overhead.
- The win requires the **draft to be much cheaper than the target**. At 0.5B vs
  1.5B-4bit that ratio is too small.

On the H200 with an fp16/bf16 **32B** target and a **3B** draft, the target step is
*enormously* more expensive than the draft step, and structured OCR output keeps
acceptance high — that's the regime where speculative decoding pays off (typically
1.5–3x). The laptop POC is for **understanding the mechanics**; the production
economics improve as the target/draft size gap grows.

---

## 6. Greedy vs. sampling (a note for later)

This POC uses **greedy** verification (accept a guess iff it equals the target's
`argmax`). It's the easiest to understand and is exactly lossless vs. greedy
target decoding.

The original papers (Leviathan et al. 2023; Chen et al. 2023) describe the
**stochastic** variant for `temperature > 0`: accept draft token `x` with
probability `min(1, p_target(x) / p_draft(x))`, and on rejection sample from the
normalized residual `max(0, p_target - p_draft)`. That scheme is provably
lossless vs. *sampling* from the target. If your OCR deployment uses sampling,
swap Step D's equality check for that probabilistic test — the rest of the loop
(propose / verify-in-one-pass / cache-crop) is unchanged.
```

# Speculative Decoding — Proof of Concept (Qwen 0.5B + 1.5B)

A laptop-scale POC to **understand speculative decoding** before deploying the
same architecture on an H200 server with larger models (Qwen 3B draft + 32B
target) for OCR document extraction.

- **Draft model**  : `Qwen/Qwen2.5-0.5B-Instruct` — fast, "guesses" tokens
- **Target model** : `Qwen/Qwen2.5-1.5B-Instruct` — accurate, "verifies" tokens
- **Task**         : OCR-style field extraction (invoice no., GSTIN, total, …)
- **Decoding**     : greedy → speculative output is **identical** to the target
  alone, just (ideally) faster (this is *lossless* speculative decoding)

> **4 GB VRAM note:** both models in fp16 (~4.1 GB) won't fit on an RTX 3050.
> The target is therefore loaded in **4-bit (NF4)** by default; the draft stays
> in fp16. The same precision is used for the "1.5B alone" baseline so the
> speedup is a fair comparison. Set `TARGET_4BIT = False` in `src/config.py`
> when you move to a big GPU.

---

## Folder structure

```
SpeculativeDecoding/
├── README.md                  # this file
├── app.py                     # Streamlit UI (streamlit run app.py)
├── requirements.txt           # Python deps (torch installed separately)
├── .gitignore
├── docs/
│   └── EXPLANATION.md         # deep-dive: every step of the algorithm
└── src/
    ├── __init__.py
    ├── config.py              # all knobs: models, precision, gamma, lengths
    ├── prompts.py             # OCR-style prompts + sample invoice + chat template
    ├── utils.py               # model loading, timing/memory, greedy baseline
    ├── speculative_decoding.py# THE core algorithm (heavily commented)
    ├── benchmark_single.py    # benchmark one model alone
    └── benchmark_compare.py   # MAIN: runs all 3 methods + prints the table
```

---

## 1. Setup (Git Bash on Windows)

```bash
cd "/d/L&T Intern/SpeculativeDecoding"

# --- create & activate a virtual environment ---
python -m venv .venv
source .venv/Scripts/activate          # Git Bash on Windows
# (PowerShell instead:  .venv\Scripts\Activate.ps1)

python -m pip install --upgrade pip
```

### 2. Install PyTorch with CUDA (do this FIRST)

Pick the index URL that matches your installed CUDA. CUDA 12.1 wheels work on
the RTX 3050 / recent drivers:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.x.x  True  NVIDIA GeForce RTX 3050 Laptop GPU
```

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

> **If `bitsandbytes` fails to install/run on Windows:** set `TARGET_4BIT = False`
> in `src/config.py`. The single-model benchmarks still work (they load one
> model at a time), but the **speculative** step needs both models resident and
> will likely OOM on 4 GB in fp16 — run that part on a GPU with ≥8 GB, or use
> 8-bit instead.

---

## 4. Run the experiment

```bash
# Benchmark each model on its own (optional sanity checks)
python -m src.benchmark_single --model draft     # Qwen 0.5B alone
python -m src.benchmark_single --model target    # Qwen 1.5B alone

# MAIN: run all three methods and print the comparison table
python -m src.benchmark_compare
```

The first run downloads the two models (~2.5 GB total) into the HuggingFace
cache; later runs are offline-fast.

### Or use the visual dashboard (Streamlit)

```bash
# optional: point at pre-downloaded local target weights
export TARGET_MODEL_ID="models/Qwen2.5-1.5B-Instruct"
streamlit run app.py
```

The UI (`app.py`) shows everything on one screen: the editable OCR document, a
task picker, and all three methods run side by side — each task's extracted
output, a **lossless check** (speculative output == 1.5B target), and the full
metrics (latency, tokens/sec, acceptance rate, accepted/rejected counts, peak
memory, speedup). γ and answer length are tunable in the sidebar. Both models
load **once** per session and stay resident (required on Windows + bitsandbytes,
where re-loading a 4-bit model segfaults).

### Measured output on this machine (RTX 3050 4 GB Laptop)

Greedy, averaged over the 5 OCR prompts, target in 4-bit:

```
| Method                  | Latency    | Tokens/sec | Acceptance Rate | Speedup          |
|-------------------------|------------|------------|-----------------|------------------|
| Qwen 0.5B alone         | ~580 ms    | 20.0       | -               | 2.14x            |
| Qwen 1.5B alone         | ~1240 ms   | 9.0        | -               | 1.00x (baseline) |
| Speculative (0.5B+1.5B) | ~1365 ms   | 7.7        | 100 %           | 0.91x            |
```

The speculative output is **token-identical** to the 1.5B alone (lossless) —
e.g. it recovers `PO-778451`, which the 0.5B draft alone gets wrong.

### Honest finding: on this 4 GB laptop, speculative decoding is *slightly slower*

Even at **100% acceptance**, the warmed, interleaved head-to-head gives
**~0.86–0.91x** (gamma sweep below). That is expected, not a bug:

- The target is **4-bit quantized** on a tiny GPU, so its per-token forward is
  already memory-bandwidth-bound and cheap-ish; speculative decoding amortizes
  the target's forward pass, but here that pass isn't expensive enough to amortize.
- The draft (0.5B fp16) costs a real fraction of the target's time, and its
  `gamma` sequential steps + the cache crop/bookkeeping add overhead each round.
- Speculative decoding wins when the **draft is much cheaper than the target**.
  At 0.5B-vs-1.5B-4bit that ratio is too small. At **3B-vs-32B in fp16 on an
  H200** the ratio is large → real speedups (typically 1.5–3x on structured
  output). This laptop POC is for *understanding the mechanics*, not for a win.

Gamma sweep (fair, heavily warmed, target-alone ≈ 1241 ms):

```
gamma=4  spec=1450 ms  accept=100.0%  speedup=0.86x
gamma=6  spec=1365 ms  accept=100.0%  speedup=0.91x   <- default
gamma=8  spec=1371 ms  accept= 95.3%  speedup=0.91x
```

`GAMMA=6` is the default in `src/config.py` (best latency at full acceptance).

Read **`docs/EXPLANATION.md`** for a step-by-step walk-through of how the draft
proposes, the target verifies, tokens are accepted/regenerated, and how the
acceptance rate and speedup are computed.

---

## 5. Tuning

All in `src/config.py`:

| Knob              | Effect                                                            |
|-------------------|------------------------------------------------------------------|
| `GAMMA`           | draft tokens proposed per round. ↑ = more potential speedup but lower acceptance. Try 3–6. |
| `MAX_NEW_TOKENS`  | answer length per prompt.                                         |
| `TARGET_4BIT`     | `True` for 4 GB laptops; `False` on big GPUs (fp16).             |
| `N_REPEATS`       | timed repetitions per prompt (averaged) to reduce noise.         |

---

## 6. Scaling to the H200 (the real goal)

The algorithm is **model-agnostic** — only `src/config.py` changes:

```python
DRAFT_MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
TARGET_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
TARGET_4BIT     = False    # H200 has 141 GB — run fp16/bf16
```

These also share the Qwen2.5 tokenizer (the hard requirement for speculative
decoding). On an H200 the larger target-vs-draft cost gap and higher acceptance
on structured OCR output typically yield **bigger** speedups than this laptop POC.
For a production server you'd then graduate from this hand-rolled loop to a
serving engine that has speculative decoding built in (e.g. vLLM), but this POC
is what makes the mechanics click.

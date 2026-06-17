"""
app.py — Streamlit UI for the speculative decoding POC
======================================================
A visual dashboard for the Qwen 0.5B (draft) + 1.5B (target) speculative
decoding experiment. It demonstrates *every* requirement of the POC in one
screen:

  * the OCR-style input document (editable),
  * the OCR extraction tasks (selectable),
  * all THREE methods run side by side:
        1. Qwen 0.5B alone   (draft baseline)
        2. Qwen 1.5B alone   (target baseline — what we want to speed up)
        3. Speculative       (0.5B drafts, 1.5B verifies),
  * the extracted output of each method per task,
  * a lossless check (speculative output == target output, token-identical),
  * the full metrics: latency, tokens/sec, acceptance rate, accepted/rejected
    token counts, peak GPU memory, and speedup vs the 1.5B target.

Run it from the project root (Git Bash on Windows):

    source .venv/Scripts/activate
    # local target weights (optional, if pre-downloaded):
    export TARGET_MODEL_ID="models/Qwen2.5-1.5B-Instruct"
    streamlit run app.py

WHY MODELS ARE LOADED ONCE (and kept resident):
On Windows + bitsandbytes a 4-bit model can only be initialized ONCE per
process — loading a second time causes a hard CUDA access violation. Streamlit
reruns this script top-to-bottom on every interaction, so we wrap loading in
@st.cache_resource: the draft + 4-bit target load a single time and stay
resident for the whole session. Both fit together in 4 GB (~2 GB weights + KV).
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src import config
from src.prompts import (
    OCR_TASKS,
    SAMPLE_DOCUMENTS,
    SAMPLE_INVOICE,
    build_chat_prompt,
    build_extract_all_prompt,
)
from src.speculative_decoding import speculative_generate
from src.utils import (
    greedy_generate,
    load_model,
    load_tokenizer,
    set_seed,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Speculative Decoding POC — Qwen 0.5B + 1.5B",
    page_icon="⚡",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Model loading (cached once per session — see module docstring)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_everything():
    """Load tokenizer + both models exactly once and keep them resident."""
    set_seed()
    tok = load_tokenizer(config.TARGET_MODEL_ID)
    # Target (4-bit) FIRST and only once; draft (fp16) is small and safe.
    target = load_model(config.TARGET_MODEL_ID, config.TARGET_4BIT)
    draft = load_model(config.DRAFT_MODEL_ID, config.DRAFT_4BIT)
    return tok, draft, target


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def _avg(results):
    """Average a list of GenResult into one dict of metrics."""
    n = len(results)
    lat = sum(r.latency_s for r in results) / n
    toks = sum(r.n_tokens for r in results) / n
    prop = sum(r.n_proposed for r in results)
    acc = sum(r.n_accepted for r in results)
    rej = sum(r.n_rejected for r in results)
    peak = max(r.peak_mem_mb for r in results)
    return {
        "latency_s": lat,
        "tokens_per_s": (toks / lat) if lat > 0 else 0.0,
        "n_tokens": toks,
        "n_proposed": prop,
        "n_accepted": acc,
        "n_rejected": rej,
        "acceptance_rate": (acc / prop) if prop else 0.0,
        "peak_mem_mb": peak,
    }


def run_all(tok, draft, target, document, tasks, gamma, max_new, n_repeats,
            prompt_mode="lite"):
    """Run the 3 methods over every selected task; return per-task + aggregates."""
    prompts = [(t, build_chat_prompt(tok, t, document, mode=prompt_mode)) for t in tasks]

    # --- Warmup (untimed): stabilize CUDA kernels for all three paths. -----
    if prompts:
        _, wp = prompts[0]
        greedy_generate(draft, tok, wp)
        greedy_generate(target, tok, wp)
        speculative_generate(draft, target, tok, wp, gamma=gamma, max_new_tokens=max_new)

    per_task = []
    draft_runs, target_runs, spec_runs = [], [], []

    progress = st.progress(0.0, text="Running benchmarks…")
    total = len(prompts) * 3
    done = 0

    for task, prompt in prompts:
        d_reps, t_reps, s_reps = [], [], []
        for _ in range(n_repeats):
            d_reps.append(greedy_generate(draft, tok, prompt))
        done += 1; progress.progress(done / total, text=f"draft: {task[:30]}…")
        for _ in range(n_repeats):
            t_reps.append(greedy_generate(target, tok, prompt))
        done += 1; progress.progress(done / total, text=f"target: {task[:30]}…")
        for _ in range(n_repeats):
            s_reps.append(speculative_generate(
                draft, target, tok, prompt, gamma=gamma, max_new_tokens=max_new))
        done += 1; progress.progress(done / total, text=f"speculative: {task[:30]}…")

        draft_runs += d_reps
        target_runs += t_reps
        spec_runs += s_reps

        d, t, s = d_reps[-1], t_reps[-1], s_reps[-1]
        per_task.append({
            "task": task,
            "draft_out": d.text.strip(),
            "target_out": t.text.strip(),
            "spec_out": s.text.strip(),
            "lossless": s.text.strip() == t.text.strip(),
            "draft_ok": d.text.strip() == t.text.strip(),
            "accept": s.acceptance_rate,
        })

    progress.empty()

    agg = {
        "draft": _avg(draft_runs),
        "target": _avg(target_runs),
        "spec": _avg(spec_runs),
    }
    return per_task, agg


def _strip_fence(txt: str) -> str:
    """Remove a leading ```json / ``` fence and trailing ``` if the model added one."""
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if "```" in t:
            t = t[:t.rfind("```")]
    return t.strip()


def run_extract_all(tok, draft, target, document, gamma, prompt_mode="lite",
                    max_new=512):
    """Extract EVERY field as one JSON via all 3 methods; return outputs + aggregates."""
    prompt = build_extract_all_prompt(tok, document, mode=prompt_mode)

    # Cheap warmup (kernels are shape-driven, so a short warmup is enough).
    greedy_generate(draft, tok, prompt, max_new_tokens=32)
    greedy_generate(target, tok, prompt, max_new_tokens=32)
    speculative_generate(draft, target, tok, prompt, gamma=gamma, max_new_tokens=32)

    progress = st.progress(0.0, text="Extracting all details (this takes ~1-2 min)…")
    dr = greedy_generate(draft, tok, prompt, max_new_tokens=max_new)
    progress.progress(1 / 3, text="0.5B draft done…")
    tr = greedy_generate(target, tok, prompt, max_new_tokens=max_new)
    progress.progress(2 / 3, text="1.5B target done…")
    sr = speculative_generate(draft, target, tok, prompt, gamma=gamma, max_new_tokens=max_new)
    progress.empty()

    agg = {"draft": _avg([dr]), "target": _avg([tr]), "spec": _avg([sr])}
    # lossless is judged on the RAW outputs; fences are stripped only for display.
    outputs = {
        "draft_json": _strip_fence(dr.text),
        "target_json": _strip_fence(tr.text),
        "spec_json": _strip_fence(sr.text),
        "lossless": sr.text.strip() == tr.text.strip(),
    }
    return outputs, agg


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.caption(
    f"Draft: `{config.DRAFT_MODEL_ID}`  \n"
    f"Target: `{config.TARGET_MODEL_ID}`  \n"
    f"Device: `{config.DEVICE}` · "
    f"Target 4-bit: `{config.TARGET_4BIT}`"
)

gamma = st.sidebar.slider(
    "γ (gamma) — draft tokens proposed per round", 1, 12, config.GAMMA,
    help="More = bigger potential speedup but lower acceptance. Default 6 here.",
)
max_new = st.sidebar.slider(
    "Max new tokens per answer", 16, 128, config.MAX_NEW_TOKENS, step=16,
)
n_repeats = st.sidebar.slider(
    "Timed repeats per task (averaged)", 1, 3, 1,
    help="More repeats = less noise, slower. A warmup pass is always run first.",
)

prompt_mode_label = st.sidebar.radio(
    "Prompt mode",
    ["Lite (laptop-tuned)", "Full (company production)"],
    index=0,
    help="Lite = compact rules that the 1.5B can follow. Full = the verbatim "
         "~2k-token company prompt (tuned for a 32B target; it overwhelms the "
         "1.5B and causes NOT FOUND on present fields).",
)
prompt_mode = "lite" if prompt_mode_label.startswith("Lite") else "full"

st.sidebar.divider()
extract_mode_label = st.sidebar.radio(
    "Extraction mode",
    ["All details (one JSON)", "Selected fields"],
    index=0,
    help="All details = the model reads the document and returns EVERY field it "
         "finds in one JSON output (not limited to a list). Selected fields = run "
         "the chosen field keywords one at a time.",
)
extract_all = extract_mode_label.startswith("All")

selected_tasks = st.sidebar.multiselect(
    "Fields (only used in 'Selected fields' mode)", OCR_TASKS, default=OCR_TASKS,
    format_func=lambda t: t.replace("Respond with only the value.", "").strip().rstrip("."),
    disabled=extract_all,
)

run = st.sidebar.button("🚀 Run all 3 methods", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚡ Speculative Decoding — Qwen 0.5B (draft) + 1.5B (target)")
st.markdown(
    "Lossless **greedy** speculative decoding on OCR-style field extraction. "
    "The draft model guesses tokens; the target verifies them in one pass. "
    "Greedy ⇒ speculative output is **token-identical** to the 1.5B alone."
)

with st.expander("📋 What this demonstrates (the POC requirements)", expanded=False):
    st.markdown(
        "- **3 methods benchmarked:** 0.5B alone, 1.5B alone, and speculative.\n"
        "- **Per method:** latency, tokens generated, tokens/sec, peak GPU memory.\n"
        "- **Speculative extras:** acceptance rate, # accepted, # rejected tokens.\n"
        "- **Speedup** of speculative vs the 1.5B target alone.\n"
        "- **OCR outputs** of every task shown side by side, with a lossless check "
        "(speculative == target).\n"
        "- All knobs (γ, length, tasks) are tunable in the sidebar — mirrors "
        "`src/config.py`."
    )

# ---------------------------------------------------------------------------
# Input document (the "OCR extracted text")
# ---------------------------------------------------------------------------
st.subheader("📄 OCR-extracted document text")
st.caption("This stands in for text produced by an OCR engine. Pick a built-in "
           "sample or paste/edit your own (real OCR output is often HTML tables).")
doc_choice = st.selectbox("Sample document", list(SAMPLE_DOCUMENTS.keys()), index=0)
# Key the text_area on the choice so switching samples loads that document,
# while still letting you edit each one independently.
document = st.text_area("Document text", value=SAMPLE_DOCUMENTS[doc_choice],
                        height=320, label_visibility="collapsed",
                        key=f"doc_{doc_choice}")

# ---------------------------------------------------------------------------
# Run + results
# ---------------------------------------------------------------------------
if run:
    with st.spinner("Loading models (first run only — cached afterwards)…"):
        tok, draft, target = load_everything()
    if extract_all:
        outputs, agg = run_extract_all(
            tok, draft, target, document, gamma, prompt_mode=prompt_mode)
        st.session_state["results"] = {"mode": "all", "outputs": outputs, "agg": agg,
                                       "gamma": gamma, "prompt_mode": prompt_mode}
    elif not selected_tasks:
        st.warning("Select at least one field in the sidebar (or switch to "
                   "'All details' mode).")
        st.stop()
    else:
        per_task, agg = run_all(
            tok, draft, target, document, selected_tasks, gamma, max_new, n_repeats,
            prompt_mode=prompt_mode)
        st.session_state["results"] = {"mode": "fields", "per_task": per_task,
                                       "agg": agg, "gamma": gamma,
                                       "prompt_mode": prompt_mode}

if "results" not in st.session_state:
    st.info("👈 Pick an **Extraction mode** and click **Run all 3 methods** in the "
            "sidebar. The first run loads the models (~10–30 s); later runs are fast.")
    st.stop()

res = st.session_state["results"]
agg = res["agg"]
d, t, s = agg["draft"], agg["target"], agg["spec"]
speedup = (t["latency_s"] / s["latency_s"]) if s["latency_s"] > 0 else 0.0

# ---- Headline metrics ------------------------------------------------------
st.subheader("📊 Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Speculative speedup vs 1.5B", f"{speedup:.2f}×",
          help="latency(target alone) / latency(speculative)")
c2.metric("Acceptance rate", f"{s['acceptance_rate']*100:.1f}%")
c3.metric("Accepted / Rejected tokens",
          f"{s['n_accepted']} / {s['n_rejected']}")
c4.metric("Peak GPU memory", f"{max(d['peak_mem_mb'], t['peak_mem_mb'], s['peak_mem_mb']):.0f} MB")

if speedup < 1.0:
    st.warning(
        f"Speculative is **{speedup:.2f}× (slightly slower)** here — expected on a "
        "4 GB laptop with a 4-bit target: the target's forward pass isn't expensive "
        "enough to amortize, and the 0.5B↔1.5B cost gap is small. The output is still "
        "**lossless**. Real speedup appears on an H200 with the 3B↔32B fp16 gap.",
        icon="ℹ️",
    )
else:
    st.success(f"Speculative achieved a **{speedup:.2f}× speedup** over the 1.5B target.")

# ---- Comparison table (the requested markdown table) -----------------------
st.markdown("#### Method comparison")
table = pd.DataFrame([
    {"Method": "Qwen 0.5B alone",
     "Latency": f"{d['latency_s']*1000:.1f} ms",
     "Tokens/sec": f"{d['tokens_per_s']:.1f}",
     "Acceptance Rate": "—",
     "Speedup": f"{t['latency_s']/d['latency_s']:.2f}×" if d['latency_s'] else "—"},
    {"Method": "Qwen 1.5B alone",
     "Latency": f"{t['latency_s']*1000:.1f} ms",
     "Tokens/sec": f"{t['tokens_per_s']:.1f}",
     "Acceptance Rate": "—",
     "Speedup": "1.00× (baseline)"},
    {"Method": "Speculative (0.5B+1.5B)",
     "Latency": f"{s['latency_s']*1000:.1f} ms",
     "Tokens/sec": f"{s['tokens_per_s']:.1f}",
     "Acceptance Rate": f"{s['acceptance_rate']*100:.1f}%",
     "Speedup": f"{speedup:.2f}×"},
])
st.table(table)
st.caption("Peak GPU memory is whole-process (both models stay resident in the UI for "
           "responsiveness), so it does not isolate per-model footprint like "
           "`src/benchmark_compare.py` does.")

# ---- Extraction outputs ----------------------------------------------------
def _field_label(task: str) -> str:
    return (task.replace("Extract the ", "")
                .replace("Respond with only the value.", "")
                .strip().rstrip("."))


if res.get("mode") == "all":
    out = res["outputs"]
    st.markdown("#### 🧾 Extracted details (full document → one JSON)")
    if out["lossless"]:
        st.success("✅ Lossless — the speculative JSON is token-identical to the "
                   "1.5B target's JSON.")
    else:
        st.warning("❌ Speculative diverged from the 1.5B target on this run.")
    st.caption("Open-ended extraction: the model returns EVERY field it finds (not "
               "limited to a predefined list). All three methods shown below; the "
               "speculative output is what you would actually deploy.")
    jc1, jc2 = st.columns(2)
    jc1.markdown("**0.5B draft**")
    jc1.code(out["draft_json"] or "(empty)", language="json")
    jc2.markdown("**1.5B target**")
    jc2.code(out["target_json"] or "(empty)", language="json")
    st.markdown("**Speculative (0.5B + 1.5B)**")
    st.code(out["spec_json"] or "(empty)", language="json")
else:
    per_task = res["per_task"]
    st.markdown("#### 🔍 OCR extraction outputs per field")
    st.caption("For each field, the value extracted by each method. ✅ lossless = the "
               "speculative output is token-identical to the 1.5B target.")
    for r in per_task:
        field = _field_label(r["task"])
        with st.container(border=True):
            head_l, head_r = st.columns([3, 1])
            head_l.markdown(f"**{field}**")
            head_r.markdown(
                f"{'✅ lossless' if r['lossless'] else '❌ diverged'} · "
                f"accept {r['accept']*100:.0f}%"
            )
            c1, c2, c3 = st.columns(3)
            c1.caption("0.5B draft")
            c1.code(r["draft_out"] or "NOT FOUND", language=None)
            c2.caption("1.5B target")
            c2.code(r["target_out"] or "NOT FOUND", language=None)
            c3.caption("Speculative (0.5B+1.5B)")
            c3.code(r["spec_out"] or "NOT FOUND", language=None)

    with st.expander("📑 All fields as one table", expanded=False):
        summary = pd.DataFrame([{
            "Field": _field_label(r["task"]),
            "0.5B draft": r["draft_out"],
            "1.5B target": r["target_out"],
            "Speculative": r["spec_out"],
            "Lossless": "✅" if r["lossless"] else "❌",
            "Draft=Target": "✅" if r["draft_ok"] else "❌",
            "Accept %": f"{r['accept']*100:.0f}%",
        } for r in per_task])
        st.table(summary)

    n_lossless = sum(1 for r in per_task if r["lossless"])
    n_draft_ok = sum(1 for r in per_task if r["draft_ok"])
    diverged = [r for r in per_task if not r["draft_ok"]]
    if diverged:
        ex = diverged[0]
        example = (f" — e.g. **{_field_label(ex['task'])}**: draft said "
                   f"`{ex['draft_out'][:30] or 'NOT FOUND'}` but speculative returned "
                   f"the target's `{ex['spec_out'][:30] or 'NOT FOUND'}`.")
    else:
        example = " — here the draft matched the target on every field."
    st.markdown(
        f"- **Lossless:** {n_lossless}/{len(per_task)} speculative outputs match the "
        f"1.5B target exactly (the correctness guarantee — independent of speed).\n"
        f"- **Draft alone correctness:** {n_draft_ok}/{len(per_task)} of the draft's "
        f"own answers matched the target{example}"
    )

# ---- Raw per-method detail -------------------------------------------------
with st.expander("🔢 Full per-method metrics", expanded=False):
    detail = pd.DataFrame([
        {"Method": "0.5B draft", **{k: round(v, 3) if isinstance(v, float) else v
                                    for k, v in d.items()}},
        {"Method": "1.5B target", **{k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in t.items()}},
        {"Method": "Speculative", **{k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in s.items()}},
    ])
    st.dataframe(detail, use_container_width=True, hide_index=True)
    if res.get("mode") == "all":
        st.caption(f"γ = {res['gamma']} · mode = extract-all (one long JSON) · "
                   f"prompt = {res.get('prompt_mode', 'lite')}.")
    else:
        st.caption(f"γ = {res['gamma']} · max_new_tokens = {max_new} · "
                   f"repeats = {n_repeats} (+1 warmup) · "
                   f"prompt = {res.get('prompt_mode', 'lite')}.")

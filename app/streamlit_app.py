"""ICD-10-CM coding with per-code confidence: Streamlit app.

Run: AWS_PROFILE=... AWS_REGION=... .venv/bin/streamlit run app/streamlit_app.py
(Bound to 127.0.0.1 by .streamlit/config.toml.)
"""
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import render  # noqa: E402
from app.ensemble import run_ensemble, summarize  # noqa: E402
from src.common.config import PROJECT_ROOT, load_config  # noqa: E402
from src.common.icd10 import load_tree  # noqa: E402

st.set_page_config(page_title="ICD-10-CM coding with confidence", layout="wide")
CFG = load_config()["app"]
LABEL = render.MODEL_LABEL


@st.cache_data
def load_examples() -> dict:
    out = {}
    for i, eid in enumerate(CFG["examples"], 1):
        path = PROJECT_ROOT / CFG["examples_dir"] / f"{eid}.json"
        if not path.exists():                       # cached examples are git-ignored; build them locally
            continue
        ex = json.loads(path.read_text())
        p = ex["patient"]
        who = f'{p["age"]} {p["age_unit"][0]}' if p.get("age") is not None else "age unknown"
        out[f'Example {i} · {(p.get("visit_reason") or "visit").lower()}, {who}'] = ex
    return out


def aws_ready() -> bool:
    return bool(os.environ.get("AWS_PROFILE") and (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")))


def progress_html(done: dict) -> str:
    def cell(m, s):
        if (m, s) not in done:
            return '<span class="cell"></span>'
        return '<span class="cell on"></span>' if done[(m, s)] else \
            '<span class="cell" style="background:#fee2e2;border-color:#fca5a5" title="error"></span>'
    cells = "".join(f'<span>{LABEL[m]}</span>' + "".join(cell(m, s) for s in CFG["seeds"]) for m in CFG["models"])
    total = len(CFG["models"]) * len(CFG["seeds"])
    return (render.CSS + f'<div class="icd"><div class="card"><div class="eyebrow">Running ensemble · '
            f'{len(done)}/{total} runs done</div><div class="grid" style="margin-top:.5rem">{cells}</div></div></div>')


def evidence_section(result: dict) -> None:
    st.markdown("##### Evidence and reasoning")
    for c in result["codes"]:
        with st.expander(f'{c["code"]} · {c["description"]} · {c["confidence"]:.0%} agreement'):
            for d in c["details"]:
                st.markdown(f'**{LABEL.get(d["model"], d["model"])}, run {d["seed"]}**')
                for q in d["evidence"]:
                    st.markdown(f"> {q}")
                for stp in d["steps"]:
                    st.markdown(f'- `{stp["code"]}` {stp["description"]}: {stp["reasoning"]}')


def method_section() -> None:
    with st.expander("How confidence works"):
        st.markdown("Each transcript is coded 9 times: Opus, Sonnet and GPT-6.1 Terra, 3 runs each, walking the "
                    "ICD-10-CM hierarchy level by level. A code's confidence is the share of runs that chose it, "
                    "weighted by model rank (1/rank). It is an agreement rate, not a probability.")
        d = PROJECT_ROOT / CFG["heldout_eval_dir"]
        if not (d / "methods.csv").exists():
            st.info("Held-out evaluation not found; run src.uncertainty_estimation.compare_methods --split heldout.")
            return
        rows = list(csv.DictReader(open(d / "methods.csv")))
        st.markdown("**Held-out evaluation (20 transcripts, full codes)**")
        st.dataframe([{"method": r["method"], "AUROC": round(float(r["auroc"]), 3),
                       "E-AURC": round(float(r["e_aurc"]), 3), "Brier": round(float(r["brier"]), 3),
                       "predicted-set exact F1": round(float(r["pred_exact_f1"]), 2)} for r in rows],
                     hide_index=True)
        st.image(str(d / "risk_coverage_full_code.png"))


st.title("ICD-10-CM coding with confidence")
st.caption("Predicted FY2025 codes for a doctor–patient transcript, each with how strongly the model ensemble agrees.")
left, main = st.columns([1, 2.6], gap="large")

with left:
    examples = load_examples()
    choice = None
    if examples:
        choice = st.radio("Examples (instant, cached)", list(examples), index=0,
                          on_change=lambda: st.session_state.pop("live", None))   # last click wins
    else:
        st.info("No cached examples. Build them with `python -m app.build_examples --runs ...` (see README).")
    if choice and not st.session_state.get("live"):
        st.markdown("**Example transcript**")
        with st.container(height=360):
            st.text(examples[choice]["transcript"])
    st.divider()
    st.markdown("**Or paste a transcript**")
    text = st.text_area("Transcript", height=220, label_visibility="collapsed")
    a, u = st.columns(2)
    age = a.number_input("Age", min_value=0, max_value=120, value=None, step=1)
    unit = u.selectbox("Unit", ["year", "month"])
    sex = st.selectbox("Sex", ["unknown", "female", "male"])
    visit = st.text_input("Visit reason (optional)")
    run = st.button("Run ensemble", type="primary", disabled=not text.strip())

with main:
    if run and not aws_ready():
        st.error("Set AWS_PROFILE and AWS_REGION in the shell that runs Streamlit, then restart it.")
    elif run:
        record = {"encounter_id": f"app-{uuid.uuid4().hex[:8]}", "age": age,
                  "age_unit": unit if age is not None else None, "sex": None if sex == "unknown" else sex,
                  "visit_reason": visit or None, "transcript": text}
        box, done, start = st.empty(), {}, time.time()
        box.html(progress_html(done))

        def on_done(model, seed, out):
            done[(model, seed)] = not out.get("error")
            box.html(progress_html(done))

        runs = run_ensemble(record, on_done)
        box.empty()
        st.session_state["live"] = {"result": summarize(load_tree(), runs, record["encounter_id"],
                                                        CFG["models"], CFG["seeds"]),
                                    "seconds": time.time() - start, "transcript": text}
    shown = st.session_state.get("live") or (
        {"result": examples[choice], "seconds": None, "transcript": examples[choice]["transcript"]} if choice else None)
    if shown is None:
        st.info("Paste a transcript and choose Run ensemble.")
        method_section()
        st.stop()
    for err in shown["result"]["errors"]:
        st.warning(err)
    st.html(render.page(shown["result"], shown["seconds"]))
    evidence_section(shown["result"])
    method_section()

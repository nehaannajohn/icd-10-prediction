"""HTML for the results view (rendered with st.html). Styling follows the evaluation-dashboard look:
slate page, white rounded cards, uppercase section labels, pill badges, one blue accent."""
from html import escape

CSS = """
<style>
.icd { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #0f172a; }
.icd .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 1rem; box-shadow: 0 1px 2px #0000000d;
             padding: 1.25rem 1.5rem; margin-bottom: 1rem; animation: fadeUp .25s ease-out both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.icd .eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #64748b; }
.icd .big { font-size: 28px; font-weight: 600; font-variant-numeric: tabular-nums; }
.icd .muted { color: #64748b; font-size: 13px; }
.icd .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin-top: .75rem; }
.icd .stat { background: #f8fafc; border-radius: .75rem; padding: .75rem 1rem; }
.icd .row { display: grid; grid-template-columns: 92px 1fr 170px 96px; gap: 1rem; align-items: center;
            padding: .7rem 0; border-top: 1px solid #f1f5f9; }
.icd .code { display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px;
             font-weight: 600; background: #eff6ff; color: #1e40af; border-radius: 999px; padding: .15rem .6rem; }
.icd .desc { font-size: 14px; } .icd .sub { font-size: 12px; color: #64748b; margin-top: 2px; }
.icd .bar { height: 8px; background: #f1f5f9; border-radius: 4px; overflow: hidden; }
.icd .bar > span { display: block; height: 100%; border-radius: 4px; }
.icd .pct { font-size: 13px; font-variant-numeric: tabular-nums; color: #334155; margin-top: 4px; }
.icd .grid { display: grid; grid-template-columns: 52px repeat(3, 12px); gap: 3px 4px; align-items: center;
             font-size: 10px; color: #64748b; }
.icd .cell { width: 12px; height: 12px; border-radius: 3px; background: #f1f5f9; border: 1px solid #e2e8f0; }
.icd .cell.on { background: #2563eb; border-color: #2563eb; }
.icd .pill { display: inline-block; font-size: 11px; font-weight: 600; border-radius: 999px; padding: .1rem .55rem;
             margin: 2px 4px 0 0; }
.icd .ok { background: #d1fae5; color: #065f46; } .icd .near { background: #fef3c7; color: #92400e; }
.icd .no { background: #fee2e2; color: #991b1b; } .icd .flag { border: 1px solid #cbd5e1; color: #475569; }
.icd .divider { display: flex; align-items: center; gap: .75rem; margin: .5rem 0; font-size: 11px; font-weight: 600;
                letter-spacing: .06em; text-transform: uppercase; color: #64748b; }
.icd .divider::before, .icd .divider::after { content: ""; flex: 1; border-top: 1px dashed #cbd5e1; }
</style>
"""

MODEL_LABEL = {"claude-opus-5-5": "Opus", "claude-sonnet-5-5": "Sonnet", "gpt-6.1-terra": "Terra"}


def _blue(conf: float) -> str:
    """Sequential single hue, light -> dark with confidence."""
    steps = ["#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
    return steps[min(int(conf * len(steps)), len(steps) - 1)]


def _vote_grid(votes: dict) -> str:
    rows = "".join(f'<span>{escape(MODEL_LABEL.get(m, m))}</span>' +
                   "".join(f'<span class="cell{" on" if v else ""}" title="run {i}: {"chose" if v else "did not choose"}">'
                           f'</span>' for i, v in enumerate(vs))
                   for m, vs in votes.items())
    return f'<div class="grid">{rows}</div>'


def _match(code: dict) -> str:
    if "in_reference" not in code:
        return ""
    if code["in_reference"]:
        return '<span class="pill ok">✓ In reference</span>'
    if code["category_match"]:
        return '<span class="pill near">≈ Same category</span>'
    return '<span class="pill no">✗ Not in reference</span>'


def _flags(code: dict) -> str:
    out = "".join(f'<span class="pill flag">Invalid code: {escape(r)}</span>' for r in code["invalid"])
    if code["excludes1_conflict_with"]:
        out += f'<span class="pill flag">Excludes1 with {escape(", ".join(code["excludes1_conflict_with"]))}</span>'
    return out


def code_row(code: dict) -> str:
    c = code["confidence"]
    return (f'<div class="row"><div><span class="code">{escape(code["code"])}</span></div>'
            f'<div><div class="desc">{escape(code["description"])}</div>'
            f'<div class="sub">{" › ".join(map(escape, code["path"]))}</div><div>{_match(code)}{_flags(code)}</div></div>'
            f'<div><div class="bar"><span style="width:{c * 100:.0f}%;background:{_blue(c)}"></span></div>'
            f'<div class="pct">{c:.0%} agreement · {code["n_votes"]}/9 runs</div></div>'
            f'<div>{_vote_grid(code["votes"])}</div></div>')


def summary_card(result: dict, seconds: float | None = None) -> str:
    codes = result["codes"]
    high = sum(c["confidence"] >= 0.5 for c in codes)
    stats = [("Codes found", len(codes)), ("Most models agree", high), ("Low agreement", len(codes) - high)]
    if "missed_reference" in result:
        hits = sum(c.get("in_reference", False) for c in codes)
        stats.append(("Matched reference", f'{hits}/{hits + len(result["missed_reference"])}'))
    elif seconds is not None:
        stats.append(("Run time", f"{seconds:.0f}s"))
    tiles = "".join(f'<div class="stat"><div class="eyebrow">{k}</div><div class="big">{v}</div></div>' for k, v in stats)
    models = " · ".join(MODEL_LABEL.get(m, m) for m in result["models"])
    return (f'<div class="card"><div class="eyebrow">Ensemble result</div>'
            f'<div class="muted">{models} × {len(result["seeds"])} runs each. Confidence = weighted share of runs '
            f'that chose the code (weights 1/rank).</div><div class="stats">{tiles}</div></div>')


def codes_card(result: dict) -> str:
    codes = result["codes"]
    high = [c for c in codes if c["confidence"] >= 0.5]
    low = [c for c in codes if c["confidence"] < 0.5]
    body = "".join(code_row(c) for c in high)
    if low:
        body += '<div class="divider">Low agreement — worth reviewing</div>' + "".join(code_row(c) for c in low)
    return f'<div class="card"><div class="eyebrow">Predicted codes, by confidence</div>{body}</div>'


def missed_card(result: dict) -> str:
    missed = result.get("missed_reference") or []
    if not missed:
        return ""
    rows = "".join(f'<div class="row"><div><span class="code">{escape(m["code"])}</span></div>'
                   f'<div class="desc">{escape(m["description"])}</div><div></div><div></div></div>' for m in missed)
    return (f'<div class="card"><div class="eyebrow">Reference codes no run predicted</div>'
            f'<div class="muted">These carry no confidence score: the ensemble never proposed them.</div>{rows}</div>')


def page(result: dict, seconds: float | None = None) -> str:
    return (CSS + '<div class="icd">' + summary_card(result, seconds) + codes_card(result) + missed_card(result)
            + "</div>")

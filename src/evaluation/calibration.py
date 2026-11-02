"""Calibration and selective-prediction metrics for per-code confidence scores.

Unit = candidate code (any code seen in >=1 run of the pool), y = 1 if it is in the reference set at that level.
"""
import random
from collections import defaultdict

import numpy as np

from src.common.icd10 import ICD10Tree


def label(tree: ICD10Tree, rows: list[dict], references: dict[str, list[str]]) -> list[dict]:
    """Add y (correct at the row's level) to each voting row."""
    ref_at = {}
    out = []
    for r in rows:
        key = (r["encounter_id"], r["level"])
        if key not in ref_at:
            ref_at[key] = {tree.ancestor(c, r["level"]) for c in references[r["encounter_id"]]}
        out.append({**r, "y": int(r["code"] in ref_at[key])})
    return out


def reliability(s: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-width bins ((m-1)/M, m/M], bin 1 includes 0. Empty bins omitted."""
    idx = np.clip(np.ceil(s * bins).astype(int), 1, bins)
    return [{"bin": m, "lo": (m - 1) / bins, "hi": m / bins, "n": int((idx == m).sum()),
             "conf": float(s[idx == m].mean()), "acc": float(y[idx == m].mean())}
            for m in range(1, bins + 1) if (idx == m).any()]


def ece(s: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    return sum(b["n"] / len(s) * abs(b["acc"] - b["conf"]) for b in reliability(s, y, bins))


def brier(s: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((s - y) ** 2))


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    """P(score of a correct code > score of an incorrect one), ties count half. NaN if one class is missing."""
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def risk_coverage(s: np.ndarray, y: np.ndarray) -> list[dict]:
    """One point per distinct score t (descending): accept {i: s_i >= t}; tied codes enter together."""
    points = []
    for t in sorted(set(s.tolist()), reverse=True):
        accepted = s >= t
        points.append({"threshold": t, "coverage": float(accepted.mean()), "risk": float(1 - y[accepted].mean())})
    return points


def aurc(s: np.ndarray, y: np.ndarray) -> float:
    """Mean over candidates i of risk(accept all with score >= s_i)."""
    risk_at = {p["threshold"]: p["risk"] for p in risk_coverage(s, y)}
    return float(np.mean([risk_at[v] for v in s.tolist()]))


def aurc_optimal(y: np.ndarray) -> float:
    """Lowest achievable AURC for these labels: correct candidates first, one at a time (no ties).
    AURC* = (1/n) * sum_{k=1..n} max(0, k - n1) / k."""
    n, n1 = len(y), int(y.sum())
    return float(np.mean([max(0, k - n1) / k for k in range(1, n + 1)])) if n else float("nan")


def e_aurc(s: np.ndarray, y: np.ndarray) -> float:
    """Excess AURC: AURC minus the best achievable on the same candidates (perfect untied ranking).
    Separates ranking quality from each method's base error rate; >= 0."""
    return aurc(s, y) - aurc_optimal(y)


def ece_floor(s: np.ndarray, bins: int = 10, n_sim: int = 2000, seed: int = 0) -> tuple[float, float]:
    """ECE expected from sampling noise alone: labels drawn as y ~ Bernoulli(s), i.e. perfectly calibrated.
    Returns (median, 95th percentile)."""
    rng = np.random.default_rng(seed)
    sims = [ece(s, (rng.random(len(s)) < s).astype(int), bins) for _ in range(n_sim)]
    return float(np.median(sims)), float(np.percentile(sims, 95))


METRICS = {"ece": ece, "brier": brier, "auroc": auroc, "aurc": aurc, "e_aurc": e_aurc}


def summarize(rows: list[dict], bins: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    """Metrics per (method, level) with 95% bootstrap CIs (resampling transcripts)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["level"])].append(r)
    rng, out = random.Random(seed), {}
    for (method, level), g in groups.items():
        s, y = np.array([r["score"] for r in g], float), np.array([r["y"] for r in g], int)
        by_enc = defaultdict(list)
        for r in g:
            by_enc[r["encounter_id"]].append((r["score"], r["y"]))
        encs = sorted(by_enc)
        boot = defaultdict(list)
        for _ in range(n_boot):
            sample = [p for e in rng.choices(encs, k=len(encs)) for p in by_enc[e]]
            bs, by = np.array([p[0] for p in sample]), np.array([p[1] for p in sample])
            for name, fn in METRICS.items():
                boot[name].append(fn(bs, by))
        floor_median, floor_p95 = ece_floor(s, bins, seed=seed)
        out[f"{method}|{level}"] = {
            "method": method, "level": level, "n": len(g), "n_correct": int(y.sum()),
            "ece_floor_median": floor_median, "ece_floor_p95": floor_p95,
            **{name: fn(s, y) for name, fn in METRICS.items()},
            **{f"{name}_ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
               for name, v in boot.items()},
            "reliability": reliability(s, y, bins), "risk_coverage": risk_coverage(s, y)}
    return out

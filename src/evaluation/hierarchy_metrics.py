"""Evaluation metrics: hierarchical (per-level counts summed over levels) and exact code (strict, full code only).

Hierarchical: every code contributes one element per level (shallow codes map to themselves at deeper levels), so an
added code the reference does not support is an FP at every level where its ancestor is not in the reference set.
Exact: predicted and reference full codes compared as sets; a near miss (F90.9 vs F90.2) is one FP and one FN.
Both are summed over records (micro) for the dataset score.
"""
from src.common.icd10 import LEVELS, ICD10Tree


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def level_counts(tree: ICD10Tree, predicted: list[str], reference: list[str]) -> dict[str, dict]:
    out = {}
    for level in LEVELS:
        p = {tree.ancestor(c, level) for c in predicted}
        g = {tree.ancestor(c, level) for c in reference}
        out[level] = prf(len(p & g), len(p - g), len(g - p))
    return out


def evaluate_record(tree: ICD10Tree, encounter_id: str, predicted: list[str], reference: list[str]) -> dict:
    levels = level_counts(tree, predicted, reference)
    hierarchical = prf(*(sum(levels[l][k] for l in LEVELS) for k in ("tp", "fp", "fn")))
    return {"encounter_id": encounter_id, "levels": levels, "hierarchical": hierarchical,
            "exact": levels["full_code"]}


def code_rows(tree: ICD10Tree, encounter_id: str, predicted: list[str], reference: list[str]) -> list[dict]:
    """Disaggregated: one row per predicted code (hit = TP at that level) and per reference code (hit = found)."""
    rows = []
    for kind, codes, other in (("predicted", predicted, reference), ("reference", reference, predicted)):
        for c in codes:
            row = {"encounter_id": encounter_id, "kind": kind, "code": c}
            for level in LEVELS:
                row[level] = tree.ancestor(c, level) in {tree.ancestor(o, level) for o in other}
            rows.append(row)
    return rows


def evaluate(tree: ICD10Tree, records: list[dict]) -> dict:
    """records: [{"encounter_id", "predicted": [codes], "reference": [codes]}]"""
    per = [evaluate_record(tree, r["encounter_id"], r["predicted"], r["reference"]) for r in records]
    total = lambda key: prf(*(sum(x[key][k] for x in per) for k in ("tp", "fp", "fn")))
    mean_f1 = lambda key: sum(x[key]["f1"] for x in per) / len(per) if per else 0.0
    return {"n_records": len(per),
            "hierarchical": total("hierarchical"), "exact": total("exact"),
            "transcript_avg_f1": {"hierarchical": mean_f1("hierarchical"), "exact": mean_f1("exact")},
            "levels": {l: prf(*(sum(x["levels"][l][k] for x in per) for k in ("tp", "fp", "fn"))) for l in LEVELS},
            "records": per}

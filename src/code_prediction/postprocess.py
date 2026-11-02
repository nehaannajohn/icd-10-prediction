"""Deterministic post-processing of predicted codes: normalize, validate, tree path, Excludes1 conflicts.
Tree search builds paths from the candidate lists, so the path is always the tree path (no mismatch check).

Codes are never dropped or repaired; problems are flagged so they stay measurable.
"""
import re
from itertools import combinations

from src.common.icd10 import ICD10Tree, normalize
from src.common.trace import Trace

_CODE = r"[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?"
_REF = re.compile(rf"({_CODE})(?:\.-)?(?:\s*-\s*({_CODE})(?:\.-)?)?")


def code_refs(note: str) -> list[tuple[str, str]]:
    """Code references cited in parentheses in a note, as (low, high) ranges.
    '(E10.-)' -> [('E10', 'E10')]; '(N18.1-N18.6)' -> [('N18.1', 'N18.6')]; '(E08-E13)' -> [('E08', 'E13')]."""
    refs = []
    for group in re.findall(r"\(([^)]*)\)", note):
        for low, high in _REF.findall(group):
            refs.append((low, high or low))
    return refs


def in_ref(code: str, low: str, high: str) -> bool:
    """True if `code` falls in the cited range; a cited code includes its descendants."""
    return code.startswith(low) or code.startswith(high) or low < code < high


def _merge(codes: list[dict]) -> list[dict]:
    """Normalize codes; merge duplicates (evidence unioned, reasoning joined, highest stated confidence)."""
    merged: dict[str, dict] = {}
    for c in codes:
        code = normalize(c["code"])
        if code not in merged:
            merged[code] = {**c, "code": code, "evidence": list(c.get("evidence", []))}
            continue
        m = merged[code]
        m["evidence"] += [e for e in c.get("evidence", []) if e not in m["evidence"]]
        m["reasoning"] = f'{m.get("reasoning", "")}\n{c.get("reasoning", "")}'.strip()
        m["stated_confidence"] = max(m.get("stated_confidence") or 0, c.get("stated_confidence") or 0)
    return list(merged.values())


def tree_path(tree: ICD10Tree, code: str) -> list[str]:
    """Chapter -> ... -> code. 7th-character codes end with the full code; invalid codes stop at the longest valid prefix."""
    stem = tree.resolve(code)
    path = [n.code for n in tree.path(stem.code)] if stem else []
    return path + [code] if path and path[-1] != code and tree.validate(code).valid else path


def excludes1_conflicts(tree: ICD10Tree, codes: list[str]) -> list[dict]:
    """Pairs where one code's own or inherited Excludes1 notes cite the other."""
    out = []
    for a, b in combinations(codes, 2):
        for x, y in ((a, b), (b, a)):
            stem = tree.resolve(x)
            notes = tree.notes(stem.code).get("excludes1", []) if stem else []
            hit = next((n for n in notes if any(in_ref(y, lo, hi) for lo, hi in code_refs(n["text"]))), None)
            if hit:
                out.append({"codes": [a, b], "note_on": x, "note_from": hit["from"], "note": hit["text"]})
                break
    return out


def postprocess(tree: ICD10Tree, codes: list[dict], trace: Trace) -> dict:
    with trace.stage("postprocess", n_in=len(codes)) as event:
        merged = _merge(codes)
        conflicts = excludes1_conflicts(tree, [c["code"] for c in merged])
        out = []
        for c in merged:
            v = tree.validate(c["code"])
            out.append({**c, "code": v.code, "valid": v.valid, "reason": v.reason, "path": tree_path(tree, v.code),
                        "excludes1_conflict_with": sorted({x for k in conflicts if v.code in k["codes"]
                                                           for x in k["codes"] if x != v.code})})
        summary = {"n_codes": len(out), "n_invalid": sum(not c["valid"] for c in out),
                   "excludes1_conflicts": conflicts}
        event.outputs.update(codes=out, summary=summary)
    return {"codes": out, "summary": summary}

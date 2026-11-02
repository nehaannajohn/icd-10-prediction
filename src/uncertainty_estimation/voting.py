"""Per-code confidence scores from pools of runs: stated confidence (U0) and weighted vote fractions (U1, U2-*).

A candidate at level l is any level-l ancestor of a code predicted in at least one run of the pool.
Vote score s_l(c) = sum_r w(r) * 1[c in ancestors_l(codes(r))] / sum_r w(r), over runs r that did not error.
"""
import json
from collections import defaultdict
from pathlib import Path

from src.common.icd10 import LEVELS, ICD10Tree


def load_runs(run_dirs: list[Path]) -> dict:
    """{model: {seed: {encounter_id: prediction output}}} from one or more run folders per model."""
    runs: dict = defaultdict(lambda: defaultdict(dict))
    for d in run_dirs:
        for f in sorted((d / "raw").glob("*_run*.json")):
            if not f.name.endswith(".trace.json"):
                out = json.loads(f.read_text())
                runs[out["model"]][out["run_seed"]][out["encounter_id"]] = out
    return runs


def method_pools(ranked_models: list[str], f1: dict[str, float], seeds: list[int], k: int) -> dict:
    """{method: [(model, seed, weight)]} for U0, U1, U2-eq, U2-rank, U2-f1. ranked_models is best-first.
    A model's weight is shared by all of its seeds."""
    top, top_k = ranked_models[0], ranked_models[:k]
    rank_w = {m: (1 / i) / sum(1 / j for j in range(1, k + 1)) for i, m in enumerate(top_k, 1)}
    f1_w = {m: f1[m] / sum(f1[x] for x in top_k) for m in top_k}
    return {"U0": [(top, seeds[0], 1.0)],
            "U1": [(top, s, 1.0) for s in seeds],
            "U2-eq": [(m, s, 1.0) for m in top_k for s in seeds],
            "U2-rank": [(m, s, rank_w[m]) for m in top_k for s in seeds],
            "U2-f1": [(m, s, f1_w[m]) for m in top_k for s in seeds]}


def _ancestors(tree: ICD10Tree, codes: list[str], level: str) -> set[str]:
    return {tree.ancestor(c, level) for c in codes}


def vote_scores(tree: ICD10Tree, runs: dict, pool: list[tuple], encounter_id: str, level: str) -> dict:
    """{node: score} for one transcript at one level, plus pool bookkeeping. Errored runs are excluded."""
    votes, total, excluded = defaultdict(float), 0.0, 0
    for model, seed, w in pool:
        out = runs[model][seed].get(encounter_id)
        if out is None or out.get("error"):
            excluded += 1
            continue
        total += w
        for node in _ancestors(tree, [c["code"] for c in out["codes"]], level):
            votes[node] += w
    return {"scores": {n: v / total for n, v in votes.items()} if total else {},
            "n_runs": len(pool) - excluded, "n_excluded": excluded}


def stated_scores(tree: ICD10Tree, runs: dict, model: str, seed: int, encounter_id: str, level: str) -> dict:
    """U0: the confidence the model stated when it selected each node during tree search.
    Uses the step whose node is the code's ancestor at `level` (deepest step for codes shallower than `level`)."""
    out = runs[model][seed].get(encounter_id)
    if out is None or out.get("error"):
        return {"scores": {}, "n_runs": 0, "n_excluded": 1}
    scores = {}
    for c in out["codes"]:
        node = tree.ancestor(c["code"], level)
        step = next((s for s in c.get("steps", []) if s["code"] == node), (c.get("steps") or [{}])[-1])
        scores[node] = step.get("confidence", c.get("stated_confidence") or 0.0)
    return {"scores": scores, "n_runs": 1, "n_excluded": 0}


def score_all(tree: ICD10Tree, runs: dict, pools: dict, encounter_ids: list[str], vote_threshold: float) -> list[dict]:
    """One row per (method, encounter, level, candidate node)."""
    rows = []
    for method, pool in pools.items():
        for eid in encounter_ids:
            for level in LEVELS:
                res = (stated_scores(tree, runs, pool[0][0], pool[0][1], eid, level) if method == "U0"
                       else vote_scores(tree, runs, pool, eid, level))
                for node, s in sorted(res["scores"].items()):
                    rows.append({"method": method, "encounter_id": eid, "level": level, "code": node, "score": s,
                                 "predicted": True if method == "U0" else s >= vote_threshold,   # U0: the run's codes
                                 "n_runs": res["n_runs"],
                                 "n_excluded": res["n_excluded"]})
    return rows

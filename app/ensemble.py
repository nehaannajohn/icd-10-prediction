"""App ensemble: top-3 models x 3 seeds tree search, U2-rank confidence per full code, and per-code detail."""
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from src.code_prediction.tree_search import predict
from src.common.artifacts import write_json
from src.common.config import load_config
from src.common.icd10 import ICD10Tree, load_tree
from src.common.trace import Trace
from src.evaluation.dataset import load_eval
from src.uncertainty_estimation.voting import method_pools, vote_scores


def run_ensemble(record: dict, on_done: Callable[[str, int, dict], None] | None = None) -> dict:
    """Run every (model, seed) in parallel. Returns {model: {seed: {encounter_id: output}}}; traces saved."""
    cfg = load_config()["app"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = load_config()["paths"]["artifacts_dir"] / f"app_{ts}"
    icl, tree, eid = load_eval("icl"), load_tree(), record["encounter_id"]
    runs = {m: {} for m in cfg["models"]}

    def one(model, seed):
        trace = Trace(run_id=run_dir.name, encounter_id=eid)
        try:
            out = {**predict(record, model, icl, run_seed=seed, trace=trace, tree=tree), "error": None}
        except Exception as e:
            out = {"encounter_id": eid, "model": model, "run_seed": seed, "codes": [], "summary": {}, "usage": {},
                   "error": f"{type(e).__name__}: {e}"}
        trace.save(run_dir / "raw" / f"{model}_run{seed}.trace.json")
        write_json(run_dir / "raw" / f"{model}_run{seed}.json", out)
        return model, seed, out

    jobs = [(m, s) for m in cfg["models"] for s in cfg["seeds"]]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for fut in as_completed([pool.submit(one, m, s) for m, s in jobs]):
            model, seed, out = fut.result()
            runs[model][seed] = {eid: out}
            if on_done:
                on_done(model, seed, out)
    return runs


def _description(tree: ICD10Tree, code: str) -> str:
    node = tree.get(code)
    return node.description if node else (f"(not a valid FY2025 code; nearest: {tree.resolve(code).code})"
                                          if tree.resolve(code) else "(not a valid FY2025 code)")


def summarize(tree: ICD10Tree, runs: dict, encounter_id: str, models: list[str], seeds: list[int],
              reference: list[str] | None = None) -> dict:
    """Display data: codes sorted by U2-rank confidence (desc), vote grid, evidence; chapter agreement;
    reference matches when a reference is given."""
    pool = method_pools(models, {m: 1.0 for m in models}, seeds, k=len(models))["U2-rank"]
    scores = vote_scores(tree, runs, pool, encounter_id, "full_code")["scores"]
    codes = []
    for code, score in scores.items():
        votes, details, flags = {}, [], {"invalid": set(), "excludes1_conflict_with": set()}
        for m in models:
            votes[m] = []
            for s in seeds:
                out = runs[m][s].get(encounter_id, {})
                hit = next((c for c in out.get("codes", []) if c["code"] == code), None)
                votes[m].append(hit is not None)
                if hit:
                    details.append({"model": m, "seed": s, "evidence": hit.get("evidence", []),
                                    "steps": [{"code": st["code"], "description": st["description"],
                                               "reasoning": st.get("reasoning", "")} for st in hit.get("steps", [])]})
                    if not hit.get("valid", True):
                        flags["invalid"].add(hit.get("reason"))
                    flags["excludes1_conflict_with"] |= set(hit.get("excludes1_conflict_with", []))
        stem = tree.resolve(code)
        codes.append({"code": code, "description": _description(tree, code), "confidence": score,
                      "votes": votes, "n_votes": sum(sum(v) for v in votes.values()),
                      "path": [n.code for n in tree.path(stem.code)] if stem else [],
                      "invalid": sorted(flags["invalid"]),
                      "excludes1_conflict_with": sorted(flags["excludes1_conflict_with"]), "details": details})
    codes.sort(key=lambda c: (-c["confidence"], c["code"]))
    chapters = vote_scores(tree, runs, pool, encounter_id, "chapter")["scores"]
    result = {"encounter_id": encounter_id, "models": models, "seeds": seeds, "codes": codes,
              "chapters": sorted(({"chapter": ch, "description": tree.get(ch).description, "agreement": a}
                                  for ch, a in chapters.items()), key=lambda x: -x["agreement"]),
              "errors": [f"{m} run {s}: {runs[m][s][encounter_id]['error']}" for m in models for s in seeds
                         if runs[m][s].get(encounter_id, {}).get("error")]}
    if reference is not None:
        ref = set(reference)
        for c in codes:
            c["in_reference"] = c["code"] in ref
            c["category_match"] = tree.ancestor(c["code"], "category") in {tree.ancestor(r, "category") for r in ref}
        result["missed_reference"] = [{"code": r, "description": _description(tree, r)}
                                      for r in reference if r not in scores]
    return result

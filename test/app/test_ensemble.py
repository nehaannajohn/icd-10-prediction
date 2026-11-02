import pytest

from app import ensemble
from src.common.icd10 import load_tree

TREE = load_tree()
MODELS, SEEDS = ["opus", "sonnet", "terra"], [0, 1, 2]


def out(codes, model="opus", seed=0, invalid=()):
    return {"encounter_id": "e", "model": model, "run_seed": seed, "error": None, "summary": {}, "usage": {},
            "codes": [{"code": c, "evidence": [f"q-{c}"], "valid": c not in invalid, "reason": "not_found",
                       "excludes1_conflict_with": [], "steps": [{"code": c, "description": "d", "reasoning": "r"}]}
                      for c in codes]}


def make_runs(table):
    return {m: {s: {"e": out(table.get((m, s), []), m, s)} for s in SEEDS} for m in MODELS}


def test_scores_votes_and_sorting():
    table = {**{("opus", s): ["E55.9", "I10"] for s in SEEDS}, ("sonnet", 0): ["E55.9"], ("sonnet", 1): ["E55.9"]}
    res = ensemble.summarize(TREE, make_runs(table), "e", MODELS, SEEDS)
    by = {c["code"]: c for c in res["codes"]}
    assert by["E55.9"]["confidence"] == pytest.approx((1 / 1) / (1 + 1 / 2 + 1 / 3) + (2 / 3) * (1 / 2) / (1 + 1 / 2 + 1 / 3))
    assert by["E55.9"]["votes"] == {"opus": [True] * 3, "sonnet": [True, True, False], "terra": [False] * 3}
    assert [c["confidence"] for c in res["codes"]] == sorted((c["confidence"] for c in res["codes"]), reverse=True)
    assert by["E55.9"]["description"].startswith("Vitamin D deficiency")
    assert {ch["chapter"] for ch in res["chapters"]} == {"Ch4", "Ch9"}


def test_invalid_flag_and_reference_matches():
    runs = make_runs({("opus", 0): ["I10", "E11.99"]})
    runs["opus"][0]["e"] = out(["I10", "E11.99"], invalid=("E11.99",))
    res = ensemble.summarize(TREE, runs, "e", MODELS, SEEDS, reference=["I10", "E11.9", "J06.9"])
    by = {c["code"]: c for c in res["codes"]}
    assert by["E11.99"]["invalid"] == ["not_found"] and "not a valid" in by["E11.99"]["description"]
    assert by["I10"]["in_reference"] and not by["E11.99"]["in_reference"] and by["E11.99"]["category_match"]
    assert [m["code"] for m in res["missed_reference"]] == ["E11.9", "J06.9"]


def test_run_ensemble_parallel_progress_and_persistence(tmp_path, monkeypatch):
    real = ensemble.load_config()
    cfg = {**real, "paths": {**real["paths"], "artifacts_dir": tmp_path},
           "app": {**real["app"], "models": MODELS, "seeds": SEEDS}}
    monkeypatch.setattr(ensemble, "load_config", lambda: cfg)
    monkeypatch.setattr(ensemble, "load_eval", lambda split=None: [])
    monkeypatch.setattr(ensemble, "predict", lambda rec, m, icl, run_seed, trace, tree: out(["I10"], m, run_seed))
    done = []
    runs = ensemble.run_ensemble({"encounter_id": "e"}, on_done=lambda m, s, o: done.append((m, s)))
    assert sorted(done) == sorted((m, s) for m in MODELS for s in SEEDS)
    assert all(runs[m][s]["e"]["codes"][0]["code"] == "I10" for m in MODELS for s in SEEDS)
    (run_dir,) = tmp_path.glob("app_*")
    assert len(list((run_dir / "raw").glob("*.trace.json"))) == 9

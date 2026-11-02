import pytest

from src.common.icd10 import load_tree
from src.uncertainty_estimation.voting import method_pools, score_all, stated_scores, vote_scores

TREE = load_tree()


def out(codes, error=None, steps=None):
    return {"codes": [{"code": c, "stated_confidence": 0.5, "steps": (steps or {}).get(c, [])} for c in codes],
            "error": error}


def runs_of(table):
    """{model: {seed: {eid: out}}}"""
    return {m: {s: {"e": o} for s, o in seeds.items()} for m, seeds in table.items()}


def test_u1_vote_fraction_over_three_seeds():
    runs = runs_of({"A": {0: out(["I10"]), 1: out(["I10"]), 2: out(["E78.5"])}})
    res = vote_scores(TREE, runs, [("A", s, 1.0) for s in range(3)], "e", "full_code")
    assert res["scores"] == {"I10": pytest.approx(2 / 3), "E78.5": pytest.approx(1 / 3)}


def test_sibling_codes_agree_at_category_level():
    runs = runs_of({"A": {0: out(["E11.9"]), 1: out(["E11.29"])}})
    pool = [("A", 0, 1.0), ("A", 1, 1.0)]
    assert vote_scores(TREE, runs, pool, "e", "category")["scores"] == {"E11": 1.0}
    assert vote_scores(TREE, runs, pool, "e", "full_code")["scores"] == {"E11.9": 0.5, "E11.29": 0.5}


def test_rank_and_f1_weights():
    pools = method_pools(["A", "B", "C", "D"], {"A": 0.59, "B": 0.54, "C": 0.48, "D": 0.45}, seeds=[0, 1, 2], k=3)
    rank_w = {m: w for m, s, w in pools["U2-rank"] if s == 0}
    assert [round(rank_w[m], 3) for m in "ABC"] == [0.545, 0.273, 0.182] and "D" not in rank_w
    f1_w = {m: w for m, s, w in pools["U2-f1"] if s == 0}
    assert f1_w["A"] == pytest.approx(0.59 / (0.59 + 0.54 + 0.48))
    assert len(pools["U1"]) == 3 and {m for m, _, _ in pools["U1"]} == {"A"} and len(pools["U2-eq"]) == 9


def test_weighted_vote_uses_model_weight_for_all_its_seeds():
    runs = runs_of({"A": {0: out(["I10"]), 1: out(["I10"])}, "B": {0: out([]), 1: out([])}})
    pool = [("A", 0, 0.75), ("A", 1, 0.75), ("B", 0, 0.25), ("B", 1, 0.25)]
    assert vote_scores(TREE, runs, pool, "e", "full_code")["scores"] == {"I10": 0.75}


def test_errored_run_excluded_from_denominator():
    runs = runs_of({"A": {0: out(["I10"]), 1: out([], error="boom"), 2: out(["I10"])}})
    res = vote_scores(TREE, runs, [("A", s, 1.0) for s in range(3)], "e", "full_code")
    assert res["scores"] == {"I10": 1.0} and (res["n_runs"], res["n_excluded"]) == (2, 1)


def test_u0_uses_confidence_stated_at_each_step():
    steps = {"E11.29": [{"code": "Ch4", "confidence": 0.9}, {"code": "E08-E13", "confidence": 0.8},
                        {"code": "E11", "confidence": 0.6}, {"code": "E11.2", "confidence": 0.55},
                        {"code": "E11.29", "confidence": 0.5}]}
    runs = runs_of({"A": {0: out(["E11.29"], steps=steps)}})
    got = {lvl: stated_scores(TREE, runs, "A", 0, "e", lvl)["scores"]
           for lvl in ("chapter", "block", "category", "subcategory_4", "subcategory_6", "full_code")}
    assert got == {"chapter": {"Ch4": 0.9}, "block": {"E08-E13": 0.8}, "category": {"E11": 0.6},
                   "subcategory_4": {"E11.2": 0.55}, "subcategory_6": {"E11.29": 0.5}, "full_code": {"E11.29": 0.5}}


def test_score_all_marks_predicted_by_threshold():
    runs = runs_of({"A": {0: out(["I10"]), 1: out(["I10", "E78.5"]), 2: out(["I10"])}})
    pools = {"U1": [("A", s, 1.0) for s in range(3)]}
    rows = [r for r in score_all(TREE, runs, pools, ["e"], 0.5) if r["level"] == "full_code"]
    assert {(r["code"], r["predicted"]) for r in rows} == {("I10", True), ("E78.5", False)}

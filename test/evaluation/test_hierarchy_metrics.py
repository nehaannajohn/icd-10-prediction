import pytest

from src.common.icd10 import LEVELS, load_tree
from src.evaluation.hierarchy_metrics import code_rows, evaluate, evaluate_record

TREE = load_tree()


def counts(d):
    return d["tp"], d["fp"], d["fn"]


def test_worked_example_from_smoke_run():
    r = evaluate_record(TREE, "ca13dccb", ["B08.1", "J30.9"], ["B08.1", "J30.89"])
    assert [counts(r["levels"][l]) for l in LEVELS] == [(2, 0, 0)] * 3 + [(1, 1, 1)] * 4
    assert counts(r["hierarchical"]) == (10, 4, 4) and r["hierarchical"]["f1"] == pytest.approx(10 / 14)
    assert counts(r["exact"]) == (1, 1, 1) and r["exact"]["f1"] == pytest.approx(0.5)


def test_unrelated_added_code_is_fp_at_every_level():
    r = evaluate_record(TREE, "e", ["B08.1", "I10"], ["B08.1"])
    assert all(r["levels"][l]["fp"] == 1 for l in LEVELS)
    assert counts(r["hierarchical"]) == (7, 7, 0) and counts(r["exact"]) == (1, 1, 0)


def test_perfect_prediction_scores_one():
    r = evaluate_record(TREE, "e", ["I10", "S69.90XA"], ["I10", "S69.90XA"])
    assert r["hierarchical"]["f1"] == 1.0 and r["exact"]["f1"] == 1.0


def test_empty_prediction_scores_zero():
    r = evaluate_record(TREE, "e", [], ["I10"])
    assert counts(r["hierarchical"]) == (0, 0, 7) and r["hierarchical"]["precision"] == 0.0
    assert r["exact"]["recall"] == 0.0


def test_dataset_totals_are_sums_of_records():
    recs = [{"encounter_id": "a", "predicted": ["B08.1", "J30.9"], "reference": ["B08.1", "J30.89"]},
            {"encounter_id": "b", "predicted": ["F90.9"], "reference": ["F90.2"]}]
    out = evaluate(TREE, recs)
    assert counts(out["hierarchical"]) == tuple(sum(counts(r["hierarchical"])[i] for r in out["records"]) for i in range(3))
    assert counts(out["exact"]) == (1, 2, 2)                          # B08.1 hit; J30.9 and F90.9 strict misses
    assert out["exact"]["f1"] < out["hierarchical"]["f1"]             # exact is the stricter metric
    assert out["levels"]["category"]["tp"] == 3


def test_code_rows_near_miss_hits_down_to_category():
    rows = code_rows(TREE, "b", ["F90.9"], ["F90.2"])
    pred = next(r for r in rows if r["kind"] == "predicted")
    assert [pred[l] for l in LEVELS] == [True, True, True, False, False, False, False]
    assert {r["kind"] for r in rows} == {"predicted", "reference"}

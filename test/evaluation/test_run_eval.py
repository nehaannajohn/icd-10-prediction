import csv
import json

import pytest

from src.evaluation.run_eval import score_run

REFS = {"a": ["B08.1", "J30.89"], "b": ["F90.2"]}


def write_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pred = lambda eid, codes, err=None: {"encounter_id": eid, "run_seed": 0, "codes": [{"code": c} for c in codes],
                                         "error": err}
    (raw / "a_run0.json").write_text(json.dumps(pred("a", ["B08.1", "J30.9"])))
    (raw / "b_run0.json").write_text(json.dumps(pred("b", [], err="ValidationException: boom")))
    (raw / "a_run0.trace.json").write_text(json.dumps({"events": []}))     # must be ignored
    return tmp_path


def test_score_run_metrics_errors_and_outputs(tmp_path):
    run = write_run(tmp_path)
    s = score_run(run, references=REFS)["seeds"][0]
    assert s["n_records"] == 2 and s["n_errors"] == 1
    a = next(r for r in s["records"] if r["encounter_id"] == "a")
    assert a["hierarchical"]["f1"] == pytest.approx(10 / 14) and a["exact"]["f1"] == pytest.approx(0.5)
    b = next(r for r in s["records"] if r["encounter_id"] == "b")
    assert b["hierarchical"]["fn"] == 7 and b["hierarchical"]["tp"] == 0        # errored run = empty prediction
    assert (run / "eval.json").exists()
    rows = list(csv.DictReader(open(run / "eval_codes.csv")))
    assert {(r["encounter_id"], r["kind"], r["code"]) for r in rows} == {
        ("a", "predicted", "B08.1"), ("a", "predicted", "J30.9"),
        ("a", "reference", "B08.1"), ("a", "reference", "J30.89"), ("b", "reference", "F90.2")}

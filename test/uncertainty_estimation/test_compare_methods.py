import csv
import json

import pytest

from src.uncertainty_estimation import compare_methods as cm

RECS = [{"encounter_id": "e1", "reference_codes": ["I10", "E78.5"]},
        {"encounter_id": "e2", "reference_codes": ["J06.9"]}]
PRED = {("A", "e1"): ["I10", "E78.5"], ("A", "e2"): ["J06.9", "R05.9"],
        ("B", "e1"): ["I10"], ("B", "e2"): ["J06.9"],
        ("C", "e1"): ["I10", "E11.9"], ("C", "e2"): ["J20.9"]}


def make_runs(tmp_path, seeds=(0, 1, 2)):
    dirs = []
    for m in "ABC":
        raw = tmp_path / f"run_{m}" / "raw"
        raw.mkdir(parents=True)
        for s in seeds:
            for e in ("e1", "e2"):
                codes = PRED[(m, e)][: 1 + (s % 2)] if s else PRED[(m, e)]       # seeds differ a little
                out = {"model": m, "run_seed": s, "encounter_id": e, "error": None,
                       "codes": [{"code": c, "stated_confidence": 0.8, "steps": []} for c in codes]}
                (raw / f"{e}_run{s}.json").write_text(json.dumps(out))
        dirs.append(raw.parent)
    sel = tmp_path / "sel"
    sel.mkdir()
    (sel / "ranking.json").write_text(json.dumps({"ranking": [
        {"model": "A", "exact_f1": 0.6}, {"model": "B", "exact_f1": 0.5}, {"model": "C", "exact_f1": 0.4}]}))
    return sel, dirs


@pytest.fixture
def patched(tmp_path, monkeypatch):
    real = cm.load_config()
    cfg = {**real, "paths": {**real["paths"], "artifacts_dir": tmp_path / "artifacts"},
           "evaluation": {**real["evaluation"], "bootstrap_resamples": 50}}
    monkeypatch.setattr(cm, "load_config", lambda: cfg)
    monkeypatch.setattr(cm, "load_eval", lambda split=None: RECS)
    return tmp_path


def test_main_writes_outputs_sorted_by_e_aurc(patched):
    sel, dirs = make_runs(patched)
    out = cm.main(["--selection", str(sel), "--runs", *map(str, dirs), "--split", "tune"])
    rows = list(csv.DictReader(open(out / "methods.csv")))
    assert {r["method"] for r in rows} == {"U0", "U1", "U2-eq", "U2-rank", "U2-f1"}
    e = [float(r["e_aurc"]) for r in rows]
    assert e == sorted(e)
    for f in ("meta.json", "metrics.json", "scores.csv", "excess_risk_coverage_full_code.png",
              "reliability_full_code.png"):
        assert (out / f).exists()


def test_missing_seed_stops_run(patched):
    sel, dirs = make_runs(patched, seeds=(0, 1))
    with pytest.raises(SystemExit, match="runs missing"):
        cm.main(["--selection", str(sel), "--runs", *map(str, dirs)])

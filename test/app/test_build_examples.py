import json

import pytest

from app import build_examples as be

MODELS, SEEDS = ["opus", "sonnet"], [0, 1]
REC = {"encounter_id": "e", "age": 30, "age_unit": "year", "sex": "male", "visit_reason": None,
       "transcript": "BP high.", "reference_codes": ["I10", "E78.5"]}


def write_runs(tmp_path, seeds=SEEDS):
    dirs = []
    for m in MODELS:
        raw = tmp_path / m / "raw"
        raw.mkdir(parents=True)
        for s in seeds:
            (raw / f"e_run{s}.json").write_text(json.dumps({"model": m, "run_seed": s, "encounter_id": "e",
                                                            "error": None, "codes": [{"code": "I10", "steps": []}]}))
        dirs.append(str(raw.parent))
    return dirs


@pytest.fixture
def patched(tmp_path, monkeypatch):
    real = be.load_config()
    cfg = {**real, "app": {**real["app"], "models": MODELS, "seeds": SEEDS, "examples": ["e"],
                           "examples_dir": str(tmp_path / "ex")}}
    monkeypatch.setattr(be, "load_config", lambda: cfg)
    monkeypatch.setattr(be, "load_eval", lambda split=None: [REC])
    return tmp_path


def test_builds_example_json(patched):
    be.main(["--runs", *write_runs(patched)])
    ex = json.loads((patched / "ex" / "e.json").read_text())
    assert ex["patient"]["age"] == 30 and ex["transcript"] == "BP high."
    assert ex["codes"][0]["code"] == "I10" and ex["codes"][0]["in_reference"]
    assert [m["code"] for m in ex["missed_reference"]] == ["E78.5"]


def test_missing_run_stops_build(patched):
    with pytest.raises(SystemExit, match="runs missing"):
        be.main(["--runs", *write_runs(patched, seeds=[0])])

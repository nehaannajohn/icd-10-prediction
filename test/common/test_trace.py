import json

import pytest

from src.common.trace import Trace


def test_stage_records_inputs_outputs_and_timing():
    trace = Trace(run_id="r1", encounter_id="e1")
    with trace.stage("tree_search.level_1", candidates=22) as event:
        event.outputs["selected"] = ["Ch4"]
    (e,) = trace.events
    assert (e.stage, e.inputs, e.outputs, e.status, e.attempt) == (
        "tree_search.level_1", {"candidates": 22}, {"selected": ["Ch4"]}, "ok", 1)
    assert e.seconds >= 0


def test_exception_marks_error_and_reraises():
    trace = Trace()
    with pytest.raises(ValueError):
        with trace.stage("parse"):
            raise ValueError("bad json")
    assert trace.events[0].status == "error"
    assert trace.events[0].error == "ValueError: bad json"


def test_retries_are_separate_events():
    trace = Trace()
    with trace.stage("llm_call", attempt=1) as e:
        e.status = "retry"
    with trace.stage("llm_call", attempt=2):
        pass
    assert [(e.attempt, e.status) for e in trace.events] == [(1, "retry"), (2, "ok")]


def test_save_writes_json(tmp_path):
    trace = Trace(run_id="r1", encounter_id="e1")
    with trace.stage("s", path=tmp_path):
        pass
    saved = json.loads(trace.save(tmp_path / "raw" / "e1.trace.json").read_text())
    assert saved["encounter_id"] == "e1" and saved["events"][0]["inputs"]["path"] == str(tmp_path)

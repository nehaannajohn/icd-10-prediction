import pytest

from src.code_prediction import prompts
from src.code_prediction import tree_search as ts
from src.common import llm
from src.common.icd10 import load_tree
from src.common.trace import Trace

TREE = load_tree()
REC = {"encounter_id": "e1", "age": 54, "age_unit": "year", "sex": "male", "visit_reason": None,
       "transcript": "BP is high. I fell on my wrist.", "reference_codes": [], "reference_descriptions": {}}


def sel(*codes):
    return {"selections": [{"parent": "?", "code": c, "reasoning": "r", "evidence": ["BP is high."],
                            "confidence": 0.9} for c in codes]}


@pytest.fixture
def scripted(monkeypatch):
    """Replace llm.call with per-step scripted replies; still runs the real check function."""
    def install(replies):
        calls = []
        def fake_call(model, system, messages, trace, stage, check=None, temperature=None):
            data = replies[len(calls)]
            calls.append(prompts.message_text(messages[0]))
            if check:
                check(data)
            return llm.LLMResult(data=data, raw="", usage={"inputTokens": 100, "outputTokens": 10})
        monkeypatch.setattr(llm, "call", fake_call)
        return calls
    return install


def test_step1_candidates_are_22_chapters():
    assert len(ts.candidates(TREE, None)) == 22


def test_seventh_character_candidates_for_leaf_that_needs_one():
    codes = [c["code"] for c in ts.candidates(TREE, [{"code": "S69.90", "path": []}])]
    assert "S69.90XA" in codes and all(len(c.replace(".", "")) == 7 for c in codes)


def test_seventh_code_pads_with_x():
    assert ts.seventh_code("S69.90", "A") == "S69.90XA"
    assert ts.seventh_code("S69.9", "D") == "S69.9XXD"


def test_check_rejects_unlisted_code_and_bad_confidence():
    check = ts.make_check(ts.candidates(TREE, None))
    check(sel("Ch9"))
    with pytest.raises(ValueError, match="not a listed candidate"):
        check(sel("Ch99"))
    with pytest.raises(ValueError, match="confidence"):
        check({"selections": [{"code": "Ch9", "confidence": 5}]})


def test_two_branches_finish_at_different_depths(scripted):
    calls = scripted([sel("Ch9", "Ch19"), sel("I10-I1A", "S60-S69"), sel("I10", "S69"),
                      sel("S69.9"), sel("S69.90"), sel("S69.90XA")])
    trace = Trace(encounter_id="e1")
    out = ts.predict(REC, "claude-sonnet-5-5", icl_examples=[], trace=trace)
    codes = {c["code"]: c for c in out["codes"]}
    assert set(codes) == {"I10", "S69.90XA"}
    assert codes["I10"]["path"] == ["Ch9", "I10-I1A", "I10"]
    assert [s["code"] for s in codes["I10"]["steps"]] == codes["I10"]["path"]      # search path == tree path
    assert codes["S69.90XA"]["path"][-1] == "S69.90XA" and codes["S69.90XA"]["valid"]
    assert out["summary"]["steps"] == 6 and out["summary"]["unfinished"] == []
    assert out["usage"] == {"inputTokens": 600, "outputTokens": 60, "cacheReadInputTokens": 0,
                            "cacheWriteInputTokens": 0}
    assert "OPEN BRANCHES" not in calls[0] and "OPEN BRANCHES" in calls[1]
    step_events = [e for e in trace.events if e.stage.startswith("tree_search.step_")]
    assert [e.outputs["finished"] for e in step_events][2] == ["I10"]


def test_unselected_open_branch_is_closed(scripted):
    scripted([sel("Ch9", "Ch19"), sel("I10-I1A"), sel("I10")])
    trace = Trace()
    out = ts.predict(REC, "claude-sonnet-5-5", icl_examples=[], trace=trace)
    step2 = next(e for e in trace.events if e.stage == "tree_search.step_2")
    assert step2.outputs["closed"] == ["Ch19"] and [c["code"] for c in out["codes"]] == ["I10"]


def test_empty_first_step_returns_no_codes(scripted):
    scripted([{"selections": []}])
    out = ts.predict(REC, "claude-sonnet-5-5", icl_examples=[])
    assert out["codes"] == [] and out["summary"]["steps"] == 1


def test_open_branches_at_max_levels_reported_unfinished(scripted, monkeypatch):
    from src.common import config as cfg
    real = cfg.load_config()
    monkeypatch.setattr(ts, "load_config", lambda: {**real, "prediction": {
        **real["prediction"], "tree_search": {"max_levels": 2}}})
    scripted([sel("Ch4"), sel("E08-E13")])
    out = ts.predict(REC, "claude-sonnet-5-5", icl_examples=[])
    assert out["codes"] == [] and out["summary"]["unfinished"] == ["E08-E13"]


def test_code_copied_with_description_is_accepted_and_cleaned(scripted):
    reply = {"selections": [{"parent": "ROOT", "code": "Ch9: Diseases of the circulatory system (I00-I99)",
                             "reasoning": "r", "evidence": [], "confidence": 0.9}]}
    ts.make_check(ts.candidates(TREE, None))(reply)                        # no error
    scripted([reply, sel("I10-I1A"), sel("I10")])
    out = ts.predict(REC, "claude-sonnet-5-5", icl_examples=[])
    assert [c["code"] for c in out["codes"]] == ["I10"]

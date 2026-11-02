from src.code_prediction.postprocess import code_refs, excludes1_conflicts, in_ref, postprocess
from src.common.icd10 import load_tree
from src.common.trace import Trace

TREE = load_tree()
FIELDS = {"code", "valid", "reason", "path", "excludes1_conflict_with"}


def test_code_refs_parses_single_dash_and_range_forms():
    assert code_refs("type 1 diabetes mellitus (E10.-)") == [("E10", "E10")]
    assert code_refs("stage (N18.1-N18.6)") == [("N18.1", "N18.6")]
    assert code_refs("diabetes (E08-E13) and gestational (O24.4-)") == [("E08", "E13"), ("O24.4", "O24.4")]
    assert code_refs("no codes here (HbA1c)") == []


def test_in_ref_includes_descendants_and_range_interior():
    assert in_ref("E10.9", "E10", "E10")
    assert in_ref("N18.30", "N18.1", "N18.6") and in_ref("N18.6", "N18.1", "N18.6")
    assert not in_ref("N18.9", "N18.1", "N18.6")


def test_every_output_code_has_all_fields():
    out = postprocess(TREE, [{"code": "I10"}, {"code": "E11.99"}, {"code": "S69.90XA"}], Trace())
    assert all(FIELDS <= set(c) for c in out["codes"])


def test_valid_code_gets_tree_path():
    (c,) = postprocess(TREE, [{"code": "e11.29"}], Trace())["codes"]
    assert c["code"] == "E11.29" and c["valid"]
    assert c["path"] == ["Ch4", "E08-E13", "E11", "E11.2", "E11.29"]


def test_seventh_character_code_path_ends_with_full_code():
    (c,) = postprocess(TREE, [{"code": "S69.90XA"}], Trace())["codes"]
    assert c["path"][-2:] == ["S69.90", "S69.90XA"]


def test_invalid_code_kept_and_flagged():
    out = postprocess(TREE, [{"code": "E11.99"}, {"code": "I10"}], Trace())
    bad = next(c for c in out["codes"] if c["code"] == "E11.99")
    assert (bad["valid"], bad["reason"], bad["path"][-1]) == (False, "not_found", "E11.9")
    assert out["summary"]["n_codes"] == 2 and out["summary"]["n_invalid"] == 1


def test_duplicates_merged():
    out = postprocess(TREE, [{"code": "I10", "evidence": ["a"], "stated_confidence": 0.6},
                             {"code": "i10", "evidence": ["a", "b"], "stated_confidence": 0.9}], Trace())
    (c,) = out["codes"]
    assert c["evidence"] == ["a", "b"] and c["stated_confidence"] == 0.9


def test_excludes1_conflict_type1_vs_type2_diabetes():
    conflicts = excludes1_conflicts(TREE, ["E11.9", "E10.9"])
    assert len(conflicts) == 1 and set(conflicts[0]["codes"]) == {"E11.9", "E10.9"}


def test_no_conflict_for_unrelated_codes():
    assert excludes1_conflicts(TREE, ["I10", "E78.5"]) == []


def test_conflict_flag_on_each_code():
    out = postprocess(TREE, [{"code": "E11.9"}, {"code": "E10.9"}, {"code": "I10"}], Trace())
    flags = {c["code"]: c["excludes1_conflict_with"] for c in out["codes"]}
    assert flags == {"E11.9": ["E10.9"], "E10.9": ["E11.9"], "I10": []}
    assert all(c["valid"] for c in out["codes"])


def test_postprocess_is_traced():
    trace = Trace()
    postprocess(TREE, [{"code": "I10"}], trace)
    assert trace.events[0].stage == "postprocess" and trace.events[0].outputs["summary"]["n_codes"] == 1

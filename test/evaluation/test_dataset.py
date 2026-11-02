import pytest

from src.common.config import load_config
from src.evaluation.dataset import assign_splits, build, parse_reference

needs_csv = pytest.mark.skipif(not load_config()["paths"]["eval_csv"].exists(), reason="eval CSV not present")


def test_parse_reference_splits_code_and_description():
    refs = parse_reference("I10 Essential hypertension\n\ne11.29 Type 2 DM with CKD\nI10 dup")
    assert refs == {"I10": "Essential hypertension", "E11.29": "Type 2 DM with CKD"}


@needs_csv
def test_build_counts_and_split_sizes():
    records = build()
    assert len(records) == 33
    assert sum(len(r["reference_codes"]) for r in records) == 120
    assert [sum(r["split"] == s for r in records) for s in ("icl", "tune", "heldout")] == [3, 10, 20]


@needs_csv
def test_split_is_deterministic():
    assert [r["split"] for r in build()] == [r["split"] for r in build()]


@needs_csv
def test_icl_covers_each_visit_type():
    icl = [r for r in build() if r["split"] == "icl"]
    codes = [r["reference_codes"] for r in icl]
    assert any(any(c.startswith("Z00") for c in cs) for cs in codes)          # preventive
    assert any(len(cs) >= 5 for cs in codes)                                  # chronic multi-code
    assert any(len(cs) <= 2 and not any(c.startswith("Z00") for c in cs) for cs in codes)  # acute


@needs_csv
def test_blank_fields_become_none():
    records = build()
    assert any(r["age"] is None and r["sex"] is None for r in records)       # 5 rows have no age/sex


def test_split_size_mismatch_raises():
    with pytest.raises(ValueError):
        assign_splits([{"encounter_id": "a", "reference_codes": []}], seed=0, n_icl=3, n_tune=10, n_heldout=20)

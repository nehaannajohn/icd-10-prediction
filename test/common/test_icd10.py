import pytest

from src.common.icd10 import LEVELS, load_tree, normalize

TREE = load_tree()


def test_counts_match_fy2025_tabular():
    levels = [n.level for n in TREE.nodes.values()]
    assert levels.count("chapter") == 22
    assert levels.count("block") == 296
    assert len(TREE.nodes) - 22 - 296 == 46498


def test_normalize():
    assert normalize(" e1129 ") == "E11.29"
    assert normalize("I10") == "I10"
    assert normalize("s6990xa") == "S69.90XA"
    assert normalize("E08-E13") == "E08-E13"


@pytest.mark.parametrize("code, expected", [
    ("E11.29",   ["Ch4",  "E08-E13", "E11", "E11.2", "E11.29", "E11.29", "E11.29"]),
    ("I10",      ["Ch9",  "I10-I1A", "I10", "I10",   "I10",    "I10",    "I10"]),
    ("S69.90XA", ["Ch19", "S60-S69", "S69", "S69.9", "S69.90", "S69.90", "S69.90XA"]),
])
def test_ancestor_at_every_level(code, expected):
    assert [TREE.ancestor(code, lvl) for lvl in LEVELS] == expected


@pytest.mark.parametrize("code, valid, reason", [
    ("E11.29",   True,  "ok"),
    ("E66.812",  True,  "ok"),                    # FY2025-only code
    ("S69.90XA", True,  "ok"),                    # 7th character with X placeholder
    ("E11",      False, "not_billable"),
    ("S69.90",   False, "seventh_char_required"),
    ("S69.90XZ", False, "invalid_seventh_char"),
    ("E11.99",   False, "not_found"),
    ("ZZZ.9",    False, "not_found"),
])
def test_validate(code, valid, reason):
    v = TREE.validate(code)
    assert (v.valid, v.reason) == (valid, reason)


def test_invalid_code_still_gets_hierarchical_ancestors():
    assert TREE.ancestor("E11.99", "category") == "E11"      # partial credit above the invalid leaf


def test_seventh_chars_inherited():
    assert "A" in TREE.seventh_chars("S69.90")


def test_notes_inherited_from_ancestors():
    notes = TREE.notes("E11.22")
    assert any("N18" in n["text"] for n in notes.get("useAdditionalCode", []))


def test_single_category_block_is_distinct_from_category():
    assert TREE.ancestor("B10.01", "block") == "B10-B10"
    assert TREE.ancestor("B10.01", "category") == "B10"

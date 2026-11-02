import random

from src.code_prediction import prompts
from src.common.icd10 import load_tree

TREE = load_tree()
REC = {"age": 2, "age_unit": "year", "sex": "female", "visit_reason": None, "transcript": "Mom says cough.",
       "reference_codes": ["Z00.129", "S69.90XA"], "reference_descriptions": {"Z00.129": "wcc", "S69.90XA": "wrist"}}
CANDS = [{"parent": "ROOT", "code": n.code, "description": n.description, "notes": {}} for n in TREE.chapters]


def test_system_prompt_states_input_and_output_format():
    for must in ("INPUT", "CANDIDATES", "OUTPUT", '"selections"', '"parent"', '"evidence"', '"confidence"'):
        assert must in prompts.SYSTEM_PROMPT


def test_system_prompt_names_no_specific_chapter():
    assert "Chapter 21" not in prompts.SYSTEM_PROMPT and "Z codes" not in prompts.SYSTEM_PROMPT


def test_encounter_block_handles_missing_fields():
    text = prompts.encounter_block({**REC, "age": None, "sex": None})
    assert "age unknown, sex unknown; visit reason: unknown" in text and "Mom says cough." in text


def test_icl_block_lists_codes_with_tree_paths():
    text = prompts.icl_block(TREE, [REC])
    assert "Z00.129" in text and "path: Ch21 >" in text
    assert "S69.90 > S69.90XA" in text                     # 7th-character code appended to stem path


def test_candidates_block_step1_lists_all_22_chapters_under_root():
    text = prompts.candidates_block(CANDS)
    assert text.count("\n  - Ch") == 22 and "Parent ROOT:" in text


def test_candidate_order_shuffled_only_with_rng():
    plain = prompts.candidates_block(CANDS)
    assert plain == prompts.candidates_block(CANDS)
    assert prompts.candidates_block(CANDS, random.Random(1)) != plain


def test_notes_shown_and_capped():
    c = [{"parent": "E11", "code": "E11.9", "description": "T2DM w/o complications",
          "notes": {"inclusionTerm": ["a", "b", "c", "d"], "excludes1": ["x (E10.-)"]}}]
    text = prompts.candidates_block(c)
    assert "includes: a; b; c |" in text and "d" not in text.split("includes:")[1].split("|")[0]
    assert "excludes1: x (E10.-)" in text


def test_build_messages_parts_and_cache_markers():
    (msg,) = prompts.build_messages("ICL", "ENC", "", "CANDIDATES\n...", step=1, level="chapter")
    kinds = ["cache" if "cache" in p else p["text"].split("\n")[0].split(":")[0] for p in msg["content"]]
    assert kinds == ["EXAMPLES", "cache", "ENCOUNTER", "cache", "STEP 1"]
    body = prompts.message_text(msg)
    assert body.index("STEP 1") < body.index("CANDIDATES") and "OPEN BRANCHES" not in body

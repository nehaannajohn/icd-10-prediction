"""Tree-search prompts: fixed system prompt, ICL / transcript / open-branch / candidate blocks, assembly.

Message order puts the parts that are identical across steps first (system, ICL, transcript)
and the step-specific parts last (open branches, candidates).
"""
import random

from src.common.icd10 import ICD10Tree

SYSTEM_PROMPT = """You are an expert medical coder assigning ICD-10-CM (FY2025) diagnosis codes to an outpatient encounter.
You work through the ICD-10-CM hierarchy one level at a time: chapter -> block -> 3-character category ->
subcategories -> (where required) 7th character. At each step you choose which candidates to pursue.

INPUT (in the user message)
1. EXAMPLES: past encounters with their correct final codes and each code's hierarchy path.
2. ENCOUNTER: patient age/sex, visit reason if known, and the full doctor-patient transcript.
3. OPEN BRANCHES (from step 2 on): codes chosen at earlier steps that still need a more specific code,
   each with its path and the reasoning given for it.
4. CANDIDATES: the only codes you may choose at this step, grouped by parent. "ROOT" is the parent at step 1.
   Some candidates show inclusion terms and Excludes1 notes.

TASK
For every parent in CANDIDATES, select zero or more of its candidates that the encounter supports.
Selecting none for a parent closes that branch.

RULES
- Choose only codes listed in CANDIDATES, copied exactly, under the parent they are listed under.
- Chapter and block steps: be inclusive. Select every chapter/block that could plausibly contain a code for
  this encounter; wrong branches are closed at later steps, but a branch you skip is lost.
- From the category step down: select the most specific candidate the transcript supports. If the
  transcript lacks the detail, select the "unspecified" candidate rather than none.
- Code conditions that are diagnosed, treated, monitored or that affect care in this encounter.
  Do not code a diagnosis described as probable, suspected, possible or "rule out"; code its symptoms instead.
- Respect Excludes1 notes (the two conditions are not coded together) and "use additional code" /
  "code first" notes (they may require selecting a companion code in another branch).
- Evidence must be verbatim quotes from the ENCOUNTER transcript.

OUTPUT
Reply with exactly one JSON object and nothing else:
{"selections": [
   {"parent": "<parent code as listed>",
    "code": "<candidate code as listed>",
    "reasoning": "<one or two sentences: why this candidate applies>",
    "evidence": ["<verbatim quote from the transcript>"],
    "confidence": <number 0-1: how confident you are this candidate is correct>}
]}
If no candidate applies anywhere, reply {"selections": []}."""


def encounter_block(record: dict) -> str:
    age = f'{record["age"]} {record["age_unit"]}' if record.get("age") is not None else "unknown"
    return (f'Patient: age {age}, sex {record.get("sex") or "unknown"}; '
            f'visit reason: {record.get("visit_reason") or "unknown"}\n'
            f'Transcript:\n{record["transcript"]}')


def icl_block(tree: ICD10Tree, examples: list[dict], rng: random.Random | None = None) -> str:
    """Examples with their correct final codes and paths; order shuffled when `rng` is given."""
    examples = list(examples)
    if rng:
        rng.shuffle(examples)
    parts = []
    for i, ex in enumerate(examples, 1):
        lines = []
        for code in ex["reference_codes"]:
            stem = tree.resolve(code)
            path = " > ".join(n.code for n in tree.path(stem.code)) if stem else code
            desc = tree.get(code).description if tree.get(code) else ex["reference_descriptions"][code]
            lines.append(f"- {code} {desc}  (path: {path}{'' if path.endswith(code) else ' > ' + code})")
        parts.append(f"EXAMPLE {i}\n{encounter_block(ex)}\nCorrect codes:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def open_branches_block(branches: list[dict]) -> str:
    """branches: [{"code", "description", "path": [{"code", "description", "reasoning"}, ...]}]"""
    if not branches:
        return ""
    lines = []
    for b in branches:
        path = " > ".join(f'{p["code"]} ({p["description"]})' for p in b["path"])
        lines.append(f'- {b["code"]}: {path}\n  reasoning so far: {b["path"][-1].get("reasoning", "")}')
    return "OPEN BRANCHES\n" + "\n".join(lines)


def _note_suffix(notes: dict, cap: int = 3) -> str:
    bits = []
    for kind, label in (("inclusionTerm", "includes"), ("excludes1", "excludes1")):
        if notes.get(kind):
            bits.append(f'{label}: {"; ".join(notes[kind][:cap])}')
    return f'  [{" | ".join(bits)}]' if bits else ""


def candidates_block(candidates: list[dict], rng: random.Random | None = None) -> str:
    """candidates: [{"parent", "code", "description", "notes"}]; grouped by parent, order shuffled within a parent when `rng`."""
    groups: dict[str, list[dict]] = {}
    for c in candidates:
        groups.setdefault(c["parent"], []).append(c)
    lines = ["CANDIDATES"]
    for parent, items in groups.items():
        items = list(items)
        if rng:
            rng.shuffle(items)
        lines.append(f"Parent {parent}:")
        lines += [f'  - {c["code"]}: {c["description"]}{_note_suffix(c.get("notes", {}))}' for c in items]
    return "\n".join(lines)


def build_messages(icl: str, encounter: str, branches: str, candidates: str, step: int, level: str) -> list[dict]:
    """One user message as text parts: fixed parts first, step-specific last. Cache markers follow the parts
    that repeat: ICL (same for every transcript in a run seed) and the encounter (same for all its steps)."""
    step_text = "\n\n".join(p for p in [f"STEP {step}: choose at the {level} level.", branches, candidates] if p)
    return [{"role": "user", "content": [
        {"text": f"EXAMPLES\n{icl}"}, {"cache": True},
        {"text": f"ENCOUNTER\n{encounter}"}, {"cache": True},
        {"text": step_text}]}]


def message_text(message: dict) -> str:
    """Flatten a message's content (str or parts) to text: for tests and readable traces."""
    c = message["content"]
    return c if isinstance(c, str) else "\n\n".join(p["text"] for p in c if "text" in p)

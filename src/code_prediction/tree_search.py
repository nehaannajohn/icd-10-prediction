"""LLM tree search over ICD-10-CM: one call per level for all open branches, up to max_levels steps, every step traced."""
import random

from src.code_prediction import prompts
from src.code_prediction.postprocess import postprocess
from src.common import llm
from src.common.config import load_config
from src.common.icd10 import LEVELS, ICD10Tree, load_tree, normalize
from src.common.trace import Trace


def seventh_code(stem: str, char: str) -> str:
    """'S69.90' + 'A' -> 'S69.90XA' (X-padded to six characters, then the 7th)."""
    return normalize(stem.replace(".", "").ljust(6, "X") + char)


def candidates(tree: ICD10Tree, open_branches: list[dict] | None) -> list[dict]:
    """Next-step candidates. Step 1 (open_branches is None): the 22 chapters under ROOT.
    Otherwise: children of each open code, or its 7th-character full codes if it is a leaf that needs one."""
    if open_branches is None:
        return [{"parent": "ROOT", "code": n.code, "description": n.description, "level": "chapter", "notes": {}}
                for n in tree.chapters]
    out = []
    for b in open_branches:
        node = tree.get(b["code"])
        if node.children:
            out += [{"parent": node.code, "code": ch.code, "description": ch.description, "level": ch.level,
                     "notes": ch.notes if ch.level not in ("chapter", "block") else {}}
                    for ch in tree.children(node.code)]
        else:
            out += [{"parent": node.code, "code": seventh_code(node.code, char), "level": "full_code",
                     "description": f"{node.description}, {meaning}", "notes": {}}
                    for char, meaning in tree.seventh_chars(node.code).items()]
    return out


def clean_code(code) -> str:
    """'Ch5: Mental, Behavioral ... (F01-F99)' -> 'Ch5'. Some models copy the whole candidate line."""
    return str(code).split(": ", 1)[0].strip()


def make_check(cands: list[dict]):
    """Reject replies that are not {"selections": [...]} with valid fields and codes from this step's candidates."""
    allowed = {c["code"] for c in cands}

    def check(data: dict) -> None:
        sel = data.get("selections")
        if not isinstance(sel, list):
            raise ValueError('reply must be {"selections": [...]}')
        problems = []
        for i, s in enumerate(sel):
            if not isinstance(s, dict) or "code" not in s:
                problems.append(f"selection {i} has no code")
            elif clean_code(s["code"]) not in allowed:
                problems.append(f'{s["code"]} is not a listed candidate (give only the code, e.g. "Ch5")')
            elif not isinstance(s.get("evidence", []), list):
                problems.append(f'{s["code"]}: evidence must be a list of quotes')
            elif not isinstance(s.get("confidence"), (int, float)) or not 0 <= s["confidence"] <= 1:
                problems.append(f'{s["code"]}: confidence must be a number between 0 and 1')
        if problems:
            raise ValueError("; ".join(problems))
    return check


def _level_label(cands: list[dict]) -> str:
    levels = sorted({c["level"] for c in cands}, key=LEVELS.index)
    return ", ".join(l.replace("full_code", "full code (7th character)") for l in levels)


def predict(record: dict, model: str, icl_examples: list[dict], run_seed: int = 0,
            trace: Trace | None = None, tree: ICD10Tree | None = None) -> dict:
    """Tree search for one transcript with one model; returns post-processed codes, summary, usage."""
    config, tree = load_config(), tree or load_tree()
    trace = trace or Trace(encounter_id=record["encounter_id"])
    rng = random.Random(run_seed) if config["prediction"]["vary_prompt_per_run"] else None
    icl, encounter = prompts.icl_block(tree, icl_examples, rng), prompts.encounter_block(record)
    open_branches, final, usage, step = None, [], dict.fromkeys(llm.USAGE_KEYS, 0), 0

    for step in range(1, config["prediction"]["tree_search"]["max_levels"] + 1):
        cands = candidates(tree, open_branches)
        if not cands:
            break
        by_code = {c["code"]: c for c in cands}
        branch_of = {b["code"]: b for b in open_branches or []}
        messages = prompts.build_messages(icl, encounter, prompts.open_branches_block(open_branches or []),
                                          prompts.candidates_block(cands, rng), step, _level_label(cands))
        with trace.stage(f"tree_search.step_{step}", model=model, run_seed=run_seed,
                         open=[b["code"] for b in open_branches or []], candidates=list(by_code)) as event:
            result = llm.call(model, prompts.SYSTEM_PROMPT, messages, trace, f"tree_search.step_{step}",
                              check=make_check(cands))
            for k in usage:
                usage[k] += result.usage.get(k, 0)
            selected = list({clean_code(s["code"]): {**s, "code": clean_code(s["code"])}
                             for s in result.data["selections"]}.values())               # clean + dedupe by code
            new_open, finished = [], []
            for s in selected:
                c = by_code[s["code"]]
                parent = branch_of.get(c["parent"])
                path = (parent["path"] if parent else []) + [{
                    "code": c["code"], "description": c["description"], "reasoning": s.get("reasoning", ""),
                    "evidence": s.get("evidence", []), "confidence": s["confidence"]}]
                if tree.validate(c["code"]).valid:
                    finished.append({"code": c["code"], "reasoning": s.get("reasoning", ""),
                                     "evidence": s.get("evidence", []),
                                     "stated_confidence": s["confidence"], "steps": path})
                else:
                    new_open.append({"code": c["code"], "description": c["description"], "path": path})
            final += finished
            closed = sorted(set(branch_of) - {by_code[s["code"]]["parent"] for s in selected})
            event.outputs.update(selected=[s["code"] for s in selected], finished=[f["code"] for f in finished],
                                 open=[b["code"] for b in new_open], closed=closed, usage=result.usage)
        open_branches = new_open
        if not open_branches:
            break

    unfinished = [b["code"] for b in open_branches or []]
    processed = postprocess(tree, final, trace)
    return {"encounter_id": record["encounter_id"], "model": model, "run_seed": run_seed,
            "codes": processed["codes"],
            "summary": {**processed["summary"], "steps": step, "unfinished": unfinished},
            "usage": usage}

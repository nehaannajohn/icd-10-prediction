"""Clean the eval CSV into one record per transcript with a list of reference codes, and a fixed split.

Run: python -m src.evaluation.dataset
Writes: paths.eval_clean (JSONL, one transcript per line), paths.eval_labels_long (CSV, one row per code).
"""
import csv
import json
import random

from src.common.config import load_config
from src.common.icd10 import load_tree, normalize


def parse_reference(text: str) -> dict[str, str]:
    """'I10 Essential hypertension\\nE78.5 Hyperlipidemia...' -> {code: EHR description}, order kept, duplicates dropped."""
    out = {}
    for line in text.splitlines():
        if line.strip():
            code, _, desc = line.strip().partition(" ")
            out.setdefault(normalize(code), desc.strip())
    return out


def _record(row: dict) -> dict:
    refs = parse_reference(row["reference_answer"])
    blank = lambda v: v.strip() or None
    return {"encounter_id": row["encounter_id"], "split": None,
            "age": int(row["age"]) if row["age"].strip() else None,
            "age_unit": blank(row["age_unit"]), "sex": blank(row["sex"]),
            "visit_reason": blank(row["visit_reason"]),
            "reference_codes": list(refs), "reference_descriptions": refs,
            "transcript": row["transcript"]}


def _icl_rules() -> dict:
    """One in-context example per visit type, chosen by rule (ties broken by seed)."""
    is_preventive = lambda r: any(c.startswith("Z00") for c in r["reference_codes"])
    return {"preventive": is_preventive,
            "chronic_multi_code": lambda r: len(r["reference_codes"]) >= 5 and not is_preventive(r),
            "acute": lambda r: len(r["reference_codes"]) <= 2 and not is_preventive(r)}


def assign_splits(records: list[dict], seed: int, n_icl: int, n_tune: int, n_heldout: int) -> list[dict]:
    if len(records) != n_icl + n_tune + n_heldout:
        raise ValueError(f"{len(records)} records but split sizes sum to {n_icl + n_tune + n_heldout}")
    rng = random.Random(seed)
    rules = _icl_rules()
    if n_icl != len(rules):
        raise ValueError(f"n_icl={n_icl} but there are {len(rules)} ICL rules")
    chosen = set()
    for rule in rules.values():
        candidates = sorted(r["encounter_id"] for r in records if rule(r) and r["encounter_id"] not in chosen)
        if not candidates:
            raise ValueError("no transcript satisfies an ICL rule")
        chosen.add(rng.choice(candidates))
    rest = sorted(r["encounter_id"] for r in records if r["encounter_id"] not in chosen)
    rng.shuffle(rest)
    split_of = {e: "icl" for e in chosen} | {e: "tune" for e in rest[:n_tune]} | {e: "heldout" for e in rest[n_tune:]}
    return [{**r, "split": split_of[r["encounter_id"]]} for r in records]


def build() -> list[dict]:
    config = load_config()
    with open(config["paths"]["eval_csv"], encoding="utf-8-sig") as f:
        records = [_record(row) for row in csv.DictReader(f)]
    tree = load_tree()
    invalid = [(r["encounter_id"], c, tree.validate(c).reason)
               for r in records for c in r["reference_codes"] if not tree.validate(c).valid]
    if invalid:
        raise ValueError(f"reference codes invalid in FY2025: {invalid}")
    s = config["split"]
    return assign_splits(records, s["seed"], s["n_icl"], s["n_tune"], s["n_heldout"])


def write(records: list[dict]) -> None:
    paths = load_config()["paths"]
    with open(paths["eval_clean"], "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    with open(paths["eval_labels_long"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encounter_id", "split", "code", "description"])
        w.writerows([r["encounter_id"], r["split"], c, d]
                    for r in records for c, d in r["reference_descriptions"].items())


def load_eval(split: str | None = None) -> list[dict]:
    """Cleaned records, optionally filtered to one split (icl | tune | heldout)."""
    with open(load_config()["paths"]["eval_clean"]) as f:
        records = [json.loads(line) for line in f]
    return [r for r in records if split is None or r["split"] == split]


if __name__ == "__main__":
    records = build()
    write(records)
    counts = {s: sum(r["split"] == s for r in records) for s in ("icl", "tune", "heldout")}
    print(f"wrote {len(records)} transcripts, {sum(len(r['reference_codes']) for r in records)} codes, splits {counts}")

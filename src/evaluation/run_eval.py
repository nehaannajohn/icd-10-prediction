"""Score a run folder: hierarchical + exact metrics per run seed; writes eval.json and eval_codes.csv.

python -m src.evaluation.run_eval --run artifacts/smoke_claude-sonnet-5-5_20261101_121545
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.common.artifacts import write_json
from src.common.icd10 import LEVELS, load_tree
from src.evaluation.dataset import load_eval
from src.evaluation.hierarchy_metrics import code_rows, evaluate


def load_predictions(run_dir: Path) -> dict[int, list[dict]]:
    """{run_seed: [prediction outputs]} from raw/*.json (trace files excluded)."""
    by_seed = defaultdict(list)
    for f in sorted((run_dir / "raw").glob("*_run*.json")):
        if not f.name.endswith(".trace.json"):
            out = json.loads(f.read_text())
            by_seed[out["run_seed"]].append(out)
    return dict(by_seed)


def score_run(run_dir: Path, references: dict[str, list[str]] | None = None) -> dict:
    tree = load_tree()
    ref = references or {r["encounter_id"]: r["reference_codes"] for r in load_eval()}
    result, rows = {"run": run_dir.name, "seeds": {}}, []
    for seed, outs in sorted(load_predictions(run_dir).items()):
        recs = [{"encounter_id": o["encounter_id"], "predicted": [c["code"] for c in o["codes"]],
                 "reference": ref[o["encounter_id"]]} for o in outs]
        result["seeds"][seed] = {**evaluate(tree, recs), "n_errors": sum(bool(o.get("error")) for o in outs)}
        rows += [{"run_seed": seed, **row} for r in recs
                 for row in code_rows(tree, r["encounter_id"], r["predicted"], r["reference"])]
    write_json(run_dir / "eval.json", result)
    with open(run_dir / "eval_codes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_seed", "encounter_id", "kind", "code", *LEVELS])
        w.writeheader()
        w.writerows(rows)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    result = score_run(p.parse_args(argv).run)
    for seed, s in result["seeds"].items():
        fmt = lambda m: f'P {m["precision"]:.2f}  R {m["recall"]:.2f}  F1 {m["f1"]:.2f}'
        print(f'seed {seed}: {s["n_records"]} records, {s["n_errors"]} errors')
        print(f'  hierarchical  {fmt(s["hierarchical"])}')
        print(f'  exact         {fmt(s["exact"])}')
        print("  per-level F1  " + "  ".join(f'{l} {s["levels"][l]["f1"]:.2f}' for l in LEVELS))
        for r in s["records"]:
            print(f'    {r["encounter_id"][:8]}  hier F1 {r["hierarchical"]["f1"]:.2f}  exact F1 {r["exact"]["f1"]:.2f}')


if __name__ == "__main__":
    main()

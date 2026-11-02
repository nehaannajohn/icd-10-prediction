"""Build cached app examples from existing held-out ensemble runs (no Bedrock calls).

python -m app.build_examples --runs artifacts/uncertainty_<model>_<ts> ...
"""
import argparse
from pathlib import Path

from app.ensemble import summarize
from src.common.artifacts import write_json
from src.common.config import PROJECT_ROOT, load_config
from src.common.icd10 import load_tree
from src.evaluation.dataset import load_eval
from src.uncertainty_estimation.voting import load_runs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, nargs="+", type=Path)
    args = p.parse_args(argv)
    cfg, tree = load_config()["app"], load_tree()
    runs, records = load_runs(args.runs), {r["encounter_id"]: r for r in load_eval()}
    out_dir = PROJECT_ROOT / cfg["examples_dir"]
    for eid in cfg["examples"]:
        rec = records[eid]
        missing = [(m, s) for m in cfg["models"] for s in cfg["seeds"] if eid not in runs.get(m, {}).get(s, {})]
        if missing:
            raise SystemExit(f"{eid}: runs missing for {missing}")
        result = summarize(tree, runs, eid, cfg["models"], cfg["seeds"], reference=rec["reference_codes"])
        write_json(out_dir / f"{eid}.json", {
            **result, "source": "held-out example (cached ensemble runs)",
            "patient": {k: rec[k] for k in ("age", "age_unit", "sex", "visit_reason")},
            "transcript": rec["transcript"], "reference": rec["reference_codes"]})
        print(f"{eid[:8]}: {len(result['codes'])} codes, {len(result['missed_reference'])} missed reference codes")


if __name__ == "__main__":
    main()

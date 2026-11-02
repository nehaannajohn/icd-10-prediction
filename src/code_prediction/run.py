"""Run tree search on an eval split and persist outputs + traces.

python -m src.code_prediction.run --model claude-sonnet-5-5 --split tune --limit 2 --seeds 0 --stage smoke
python -m src.code_prediction.run --model claude-opus-5-5 --split tune --seeds 1 2 --stage uncertainty
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from src.code_prediction.tree_search import predict
from src.common.artifacts import new_run_dir, write_json
from src.common.icd10 import load_tree
from src.common.llm import USAGE_KEYS
from src.common.trace import Trace
from src.evaluation.dataset import load_eval

STAGES = ["smoke", "prediction", "uncertainty"]   # first part of the run folder name: artifacts/<stage>_<model>_<ts>/


def run_one(record: dict, model: str, icl: list[dict], run_seed: int, run_dir) -> dict:
    trace = Trace(run_id=run_dir.name, encounter_id=record["encounter_id"])
    name = f'{record["encounter_id"]}_run{run_seed}'
    try:
        out = predict(record, model, icl, run_seed=run_seed, trace=trace, tree=load_tree())
        out["error"] = None
    except Exception as e:
        out = {"encounter_id": record["encounter_id"], "model": model, "run_seed": run_seed, "codes": [],
               "summary": {}, "usage": {}, "error": f"{type(e).__name__}: {e}"}
    finally:
        trace.save(run_dir / "raw" / f"{name}.trace.json")
    write_json(run_dir / "raw" / f"{name}.json", out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="tune", choices=["tune", "heldout"])
    p.add_argument("--limit", type=int, default=None, help="first N transcripts of the split")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="run seeds; each seed is one run per transcript")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stage", default="prediction", choices=STAGES)
    args = p.parse_args(argv)

    records = load_eval(args.split)[: args.limit]
    icl = load_eval("icl")
    run_dir = new_run_dir(args.stage, args.model, {"args": vars(args),
                                                   "encounter_ids": [r["encounter_id"] for r in records]})
    jobs = [(r, s) for r in records for s in args.seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outs = list(pool.map(lambda j: run_one(j[0], args.model, icl, j[1], run_dir), jobs))

    ref = {r["encounter_id"]: set(r["reference_codes"]) for r in records}
    per_run = []
    for o in outs:
        pred = {c["code"] for c in o["codes"]}
        gold = ref[o["encounter_id"]]
        per_run.append({"encounter_id": o["encounter_id"], "run_seed": o["run_seed"], "error": o["error"],
                        "predicted": sorted(pred), "reference": sorted(gold), "matched": sorted(pred & gold),
                        "steps": o["summary"].get("steps"), "unfinished": o["summary"].get("unfinished"),
                        "n_invalid": o["summary"].get("n_invalid"), "usage": o["usage"]})
    totals = {k: sum(r["usage"].get(k, 0) for r in per_run) for k in USAGE_KEYS}
    totals["totalInputTokens"] = (totals["inputTokens"] + totals["cacheReadInputTokens"]
                                  + totals["cacheWriteInputTokens"])          # Bedrock: inputTokens excludes cached
    write_json(run_dir / "summary.json", {"n_runs": len(per_run), "n_errors": sum(bool(r["error"]) for r in per_run),
                                          "tokens": totals, "runs": per_run})
    print(json.dumps({"run_dir": str(run_dir), "tokens": totals,
                      "errors": sum(bool(r["error"]) for r in per_run)}, indent=2))
    for r in per_run:
        print(f'{r["encounter_id"][:8]} run{r["run_seed"]}: matched {len(r["matched"])}/{len(r["reference"])} '
              f'ref, predicted {len(r["predicted"])}, steps {r["steps"]}, error {r["error"]}')


if __name__ == "__main__":
    main()

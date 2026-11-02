"""Rank models from their prediction runs: exact micro-F1, ties broken by hierarchical micro-F1.

python -m src.evaluation.compare_runs artifacts/prediction_<model>_<ts> [...]
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

from src.common.artifacts import write_json
from src.common.config import load_config
from src.evaluation.run_eval import score_run


def row_for(run_dir: Path) -> dict:
    ev = json.loads((run_dir / "eval.json").read_text()) if (run_dir / "eval.json").exists() else score_run(run_dir)
    s = ev["seeds"]["0"] if "0" in ev["seeds"] else ev["seeds"][0]
    summary = json.loads((run_dir / "summary.json").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    return {"model": meta["model"], "run": run_dir.name, "n_records": s["n_records"], "n_errors": s["n_errors"],
            "exact_p": s["exact"]["precision"], "exact_r": s["exact"]["recall"], "exact_f1": s["exact"]["f1"],
            "hier_p": s["hierarchical"]["precision"], "hier_r": s["hierarchical"]["recall"],
            "hier_f1": s["hierarchical"]["f1"],
            **{f"f1_{l}": v["f1"] for l, v in s["levels"].items()},
            "n_invalid": sum(r.get("n_invalid") or 0 for r in summary["runs"]),
            "total_input_tokens": summary["tokens"].get("totalInputTokens"),
            "output_tokens": summary["tokens"].get("outputTokens")}


def rank(rows: list[dict]) -> list[dict]:
    ranked = sorted(rows, key=lambda r: (-r["exact_f1"], -r["hier_f1"]))
    return [{"rank": i, **r} for i, r in enumerate(ranked, 1)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    ranked = rank([row_for(r) for r in p.parse_args(argv).runs])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = load_config()["paths"]["artifacts_dir"] / f"model_selection_{ts}"
    write_json(out / "ranking.json", {"rule": "exact micro-F1, ties by hierarchical micro-F1", "ranking": ranked})
    with open(out / "ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0]))
        w.writeheader()
        w.writerows(ranked)
    print(f"{'#':>2} {'model':18} {'exact P/R/F1':>18} {'hier P/R/F1':>18} {'err':>3} {'invalid':>7} {'input tok':>10}")
    for r in ranked:
        print(f'{r["rank"]:>2} {r["model"]:18} {r["exact_p"]:.2f}/{r["exact_r"]:.2f}/{r["exact_f1"]:.2f}'
              f'     {r["hier_p"]:.2f}/{r["hier_r"]:.2f}/{r["hier_f1"]:.2f}  {r["n_errors"]:>3} {r["n_invalid"]:>7}'
              f' {r["total_input_tokens"]:>10}')
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

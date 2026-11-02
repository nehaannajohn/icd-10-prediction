"""Compare uncertainty methods (U0, U1, U2-eq/rank/f1) on a split: calibration + selective-prediction metrics,
and the quality of each method's predicted set.

python -m src.uncertainty_estimation.compare_methods --selection artifacts/model_selection_<ts> \
    --runs artifacts/prediction_<m>_<ts> ... artifacts/uncertainty_<m>_<ts> ... --split tune
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

from src.common.artifacts import git_commit, write_json
from src.common.config import load_config
from src.common.icd10 import load_tree
from src.evaluation.calibration import label, summarize
from src.evaluation.dataset import load_eval
from src.evaluation.hierarchy_metrics import evaluate
from src.uncertainty_estimation import plots
from src.uncertainty_estimation.voting import load_runs, method_pools, score_all


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True, type=Path, help="model_selection_<ts> folder (ranking.json)")
    p.add_argument("--runs", required=True, nargs="+", type=Path, help="prediction/uncertainty run folders")
    p.add_argument("--split", default="tune", choices=["tune", "heldout"])
    args = p.parse_args(argv)
    config, tree = load_config(), load_tree()
    u, ev = config["uncertainty"], config["evaluation"]

    ranking = json.loads((args.selection / "ranking.json").read_text())["ranking"]
    ranked, f1 = [r["model"] for r in ranking], {r["model"]: r["exact_f1"] for r in ranking}
    seeds = list(range(u["n_runs"]))
    pools = method_pools(ranked, f1, seeds, u["top_k"])
    records = load_eval(args.split)
    eids, refs = [r["encounter_id"] for r in records], {r["encounter_id"]: r["reference_codes"] for r in records}
    runs = load_runs(args.runs)
    missing = sorted({(m, s) for pool in pools.values() for m, s, _ in pool
                      if any(e not in runs.get(m, {}).get(s, {}) for e in eids)})
    if missing:
        raise SystemExit(f"runs missing for (model, seed): {missing}")

    rows = label(tree, score_all(tree, runs, pools, eids, u["vote_threshold"]), refs)
    metrics = summarize(rows, ev["calibration_bins"], ev["bootstrap_resamples"], seed=config["split"]["seed"])

    table = []
    for method in pools:
        full = [r for r in rows if r["method"] == method and r["level"] == "full_code"]
        pred = evaluate(tree, [{"encounter_id": e, "reference": refs[e],
                                "predicted": [r["code"] for r in full if r["encounter_id"] == e and r["predicted"]]}
                               for e in eids])
        m = metrics[f"{method}|full_code"]
        table.append({"method": method, "candidates": m["n"], "correct": m["n_correct"],
                      "ece": m["ece"], "ece_floor": m["ece_floor_median"], "brier": m["brier"], "auroc": m["auroc"], "aurc": m["aurc"],
                      "e_aurc": m["e_aurc"], "e_aurc_lo": m["e_aurc_ci"][0], "e_aurc_hi": m["e_aurc_ci"][1],
                      "pred_exact_p": pred["exact"]["precision"], "pred_exact_r": pred["exact"]["recall"],
                      "pred_exact_f1": pred["exact"]["f1"], "pred_hier_f1": pred["hierarchical"]["f1"]})
    table.sort(key=lambda t: t["e_aurc"])

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config["paths"]["artifacts_dir"] / f"uncertainty_eval_{args.split}_{ts}"
    write_json(out / "meta.json", {"split": args.split, "timestamp": ts, "git_commit": git_commit(),
                                   "selection": str(args.selection), "runs": [str(r) for r in args.runs],
                                   "pools": pools, "vote_threshold": u["vote_threshold"]})
    write_json(out / "metrics.json", metrics)
    for name, data in (("methods.csv", table), ("scores.csv", rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    plots.risk_coverage(metrics, list(pools), "full_code", out / "risk_coverage_full_code.png")
    plots.reliability(metrics, list(pools), "full_code", out / "reliability_full_code.png")

    print(f"{'method':8} {'cand':>5} {'corr':>5} {'ECE (chance)':>12} {'Brier':>6} {'AUROC':>6} {'AURC':>5} "
          f"{'E-AURC [95% CI]':>20} {'pred exact P/R/F1':>18} {'hier F1':>7}")
    for t in table:
        print(f'{t["method"]:8} {t["candidates"]:>5} {t["correct"]:>5} {t["ece"]:.3f} ({t["ece_floor"]:.3f}) {t["brier"]:.3f}  '
              f'{t["auroc"]:.3f} {t["aurc"]:.3f}  {t["e_aurc"]:.3f} [{t["e_aurc_lo"]:.3f}, {t["e_aurc_hi"]:.3f}]  '
              f'{t["pred_exact_p"]:.2f}/{t["pred_exact_r"]:.2f}/{t["pred_exact_f1"]:.2f}   {t["pred_hier_f1"]:.2f}')
    print(f"saved to {out}")
    return out


if __name__ == "__main__":
    main()

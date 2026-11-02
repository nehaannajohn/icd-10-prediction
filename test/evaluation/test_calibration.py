import numpy as np
import pytest

from src.common.icd10 import load_tree
from src.evaluation import calibration as cal

TREE = load_tree()


def test_perfect_scores():
    s, y = np.array([1.0, 1.0, 0.0]), np.array([1, 1, 0])
    assert cal.ece(s, y) == pytest.approx(0) and cal.brier(s, y) == 0
    assert cal.auroc(s, y) == 1.0
    assert cal.aurc(s, y) == pytest.approx(1 / 9)          # floor: risk at full coverage = base error rate
    assert cal.e_aurc(s, y) == pytest.approx(0)            # perfect ranking -> no excess


def test_e_aurc_zero_for_untied_perfect_ranking_and_never_negative():
    y = np.array([1, 0, 0])
    assert cal.aurc_optimal(y) == pytest.approx((0 + 1 / 2 + 2 / 3) / 3)
    assert cal.e_aurc(np.array([1.0, 0.5, 0.2]), y) == pytest.approx(0)       # was -0.056 with the tied oracle
    assert cal.e_aurc(np.array([1.0, 0.0, 0.0]), y) > 0                       # tied wrong codes cost extra


def test_ece_floor_reproducible_and_brackets_calibrated_ece():
    rng = np.random.default_rng(1)
    s = rng.choice([0.2, 0.5, 0.8], size=80)
    median, p95 = cal.ece_floor(s, seed=7)
    assert (median, p95) == cal.ece_floor(s, seed=7) and 0 < median < p95
    hits = sum(cal.ece(s, (rng.random(80) < s).astype(int)) <= p95 for _ in range(200))
    assert hits >= 170                                                        # ~95% of calibrated draws below p95


def test_e_aurc_positive_for_inverted_ranking():
    s, y = np.array([0.0, 0.0, 1.0]), np.array([1, 1, 0])
    assert cal.e_aurc(s, y) > 0


def test_all_tied_auroc_is_half():
    assert cal.auroc(np.array([0.5, 0.5, 0.5]), np.array([1, 0, 1])) == 0.5


def test_ece_hand_computed():
    # bin 9: conf 0.9, acc 0.5, n=2; bin 2: conf 0.2, acc 0, n=1 -> 2/3*0.4 + 1/3*0.2 = 0.333
    assert cal.ece(np.array([0.9, 0.9, 0.2]), np.array([1, 0, 0])) == pytest.approx(1 / 3)


def test_aurc_hand_computed_with_ties():
    # risk at t=1.0 is 0; at t=0.5 (both tied codes enter) is 1/3 -> (0 + 1/3 + 1/3) / 3
    assert cal.aurc(np.array([1.0, 0.5, 0.5]), np.array([1, 1, 0])) == pytest.approx(2 / 9)


def test_label_is_level_aware():
    rows = [{"method": "U1", "encounter_id": "e", "level": "category", "code": "E11", "score": 1.0},
            {"method": "U1", "encounter_id": "e", "level": "full_code", "code": "E11.9", "score": 1.0}]
    ys = [r["y"] for r in cal.label(TREE, rows, {"e": ["E11.29"]})]
    assert ys == [1, 0]


def test_summarize_bootstrap_ci_contains_estimate_and_is_reproducible():
    rows = [{"method": "U1", "level": "full_code", "encounter_id": f"e{i % 5}", "code": f"c{i}",
             "score": s, "y": y} for i, (s, y) in enumerate(zip([1, 1, .67, .67, .33, .33, 1, .33, .67, 1],
                                                                [1, 1, 1, 0, 0, 0, 1, 1, 0, 1]))]
    a = cal.summarize(rows, n_boot=200, seed=3)["U1|full_code"]
    b = cal.summarize(rows, n_boot=200, seed=3)["U1|full_code"]
    assert a["aurc_ci"] == b["aurc_ci"]
    assert a["aurc_ci"][0] <= a["aurc"] <= a["aurc_ci"][1]
    assert a["n"] == 10 and sum(x["n"] for x in a["reliability"]) == 10

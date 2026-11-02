from src.evaluation.compare_runs import rank


def test_rank_by_exact_then_hierarchical():
    rows = [{"model": "a", "exact_f1": 0.4, "hier_f1": 0.9}, {"model": "b", "exact_f1": 0.5, "hier_f1": 0.1},
            {"model": "c", "exact_f1": 0.4, "hier_f1": 0.95}]
    assert [r["model"] for r in rank(rows)] == ["b", "c", "a"]

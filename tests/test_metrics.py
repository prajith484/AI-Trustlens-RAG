from evaluation.metrics import coverage, precision_at_k, recall_at_k, reciprocal_rank


def test_metrics():
    assert recall_at_k(["a", "b"], ["a", "c"], 2) == 0.5
    assert precision_at_k(["a", "x"], ["a", "c"], 2) == 0.5
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0
    assert recall_at_k(["a"], [], 5) == 0.0
    assert coverage([{"hits": 1}, {"hits": 0}]) == 0.5

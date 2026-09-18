import pytest

from evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_rate_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c"}

    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 2) == 0.0


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 4) == 0.5


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}

    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_reciprocal_rank():
    retrieved = ["x", "y", "z"]
    relevant = {"y"}

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_when_missing():
    retrieved = ["x", "y", "z"]
    relevant = {"missing"}

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_mean_reciprocal_rank():
    rankings = [
        ["a", "b", "c"],
        ["x", "y", "z"],
    ]

    relevant_sets = [
        {"a"},
        {"z"},
    ]

    assert mean_reciprocal_rank(rankings, relevant_sets) == pytest.approx(
        (1.0 + 1.0 / 3.0) / 2
    )


def test_ndcg_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}

    score = ndcg_at_k(retrieved, relevant, 3)

    assert 0.0 < score <= 1.0


def test_ndcg_perfect_ranking():
    retrieved = ["a", "c", "x"]
    relevant = {"a", "c"}

    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_empty_relevant_set():
    retrieved = ["a", "b"]

    assert hit_rate_at_k(retrieved, set(), 2) == 0.0
    assert precision_at_k(retrieved, set(), 2) == 0.0
    assert recall_at_k(retrieved, set(), 2) == 0.0
    assert reciprocal_rank(retrieved, set()) == 0.0
    assert ndcg_at_k(retrieved, set(), 2) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        hit_rate_at_k(["a"], {"a"}, 0)

    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)

    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 0)

    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, 0)
from evaluation.graph_metrics import (
    calculate_graph_hit_rate_at_k,
    calculate_graph_recall_at_k,
    calculate_graph_mrr,
    calculate_relationship_accuracy,
)


def test_graph_hit_rate():
    retrieved = ["func_a", "func_b", "func_c"]
    expected = ["func_b", "func_z"]
    assert calculate_graph_hit_rate_at_k(retrieved, expected, k=2) == 1.0
    assert calculate_graph_hit_rate_at_k(retrieved, ["unknown"], k=2) == 0.0


def test_graph_recall():
    retrieved = ["func_a", "func_b"]
    expected = ["func_a", "func_b", "func_c", "func_d"]
    assert calculate_graph_recall_at_k(retrieved, expected, k=2) == 0.5


def test_graph_mrr():
    retrieved = ["func_x", "func_target", "func_y"]
    expected = ["func_target"]
    assert calculate_graph_mrr(retrieved, expected) == 0.5


def test_relationship_accuracy():
    retrieved = [
        ("A", "CALLS", "B"),
        ("B", "CALLS", "C"),
        ("C", "IMPORTS", "D"),
    ]
    expected = [
        ("A", "CALLS", "B"),
        ("B", "CALLS", "C"),
    ]
    assert calculate_relationship_accuracy(retrieved, expected) == 1.0
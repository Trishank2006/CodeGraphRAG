from evaluation.path_metrics import evaluate_path_reconstruction


def test_path_exact_match():
    path = ["login", "validate", "db_query"]
    res = evaluate_path_reconstruction(path, path)
    assert res["exact_match"] is True
    assert res["node_accuracy"] == 1.0
    assert res["length_difference"] == 0


def test_path_partial_match():
    retrieved = ["login", "cache", "db_query"]
    expected = ["login", "validate", "db_query"]
    res = evaluate_path_reconstruction(retrieved, expected)
    assert res["exact_match"] is False
    assert res["node_accuracy"] == 2.0 / 3.0
    assert res["length_difference"] == 0


def test_missing_path():
    res = evaluate_path_reconstruction([], ["login", "validate"])
    assert res["exact_match"] is False
    assert res["node_accuracy"] == 0.0
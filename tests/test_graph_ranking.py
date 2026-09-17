from graph.ranking import calculate_graph_score, RELATION_WEIGHTS


def test_relation_weights_decay():
    # Direct CALLS
    score_calls_1 = calculate_graph_score("CALLS", distance=1)
    assert score_calls_1 == 1.0

    # 2-hop CALLS (1.0 * 0.7 = 0.7)
    score_calls_2 = calculate_graph_score("CALLS", distance=2)
    assert score_calls_2 == 0.7

    # Direct IMPORTS (0.8 * 1.0 = 0.8)
    score_imports_1 = calculate_graph_score("IMPORTS", distance=1)
    assert score_imports_1 == 0.8

    # 2-hop IMPORTS (0.8 * 0.7 = 0.56)
    score_imports_2 = calculate_graph_score("IMPORTS", distance=2)
    assert score_imports_2 == 0.56


def test_unknown_relation_uses_default():
    score = calculate_graph_score("UNKNOWN_EDGE", distance=1)
    assert score == 0.6
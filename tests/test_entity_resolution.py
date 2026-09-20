from unittest.mock import MagicMock
from evaluation.entity_resolution import EntityResolutionEvaluator


def test_entity_resolution():
    mock_queries = MagicMock()
    mock_queries.find_entities.return_value = [
        {"id": "auth.py:func:login", "repository": "repo_a"}
    ]

    evaluator = EntityResolutionEvaluator(mock_queries)
    res = evaluator.evaluate_symbol_lookup("login", "auth.py:func:login", repository="repo_a")

    assert res["found"] is True
    assert res["is_correct"] is True
    assert res["is_isolated"] is True
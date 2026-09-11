from unittest.mock import MagicMock
from graph.queries import GraphQueries


def test_get_function_callers():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {"id": "file:service.py:function:checkout", "name": "checkout", "file_path": "service.py"}
    ]

    queries = GraphQueries(store=mock_store)
    callers = queries.get_function_callers("process_payment")

    assert len(callers) == 1
    assert callers[0]["name"] == "checkout"
    mock_store.execute_query.assert_called_once()
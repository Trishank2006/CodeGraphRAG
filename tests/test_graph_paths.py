from unittest.mock import MagicMock
from graph.path import GraphPathFinder


def test_find_shortest_path():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "nodes": [
                {"id": "1", "name": "OrderController", "label": "Class"},
                {"id": "2", "name": "create_order", "label": "Function"},
                {"id": "3", "name": "process_payment", "label": "Function"},
            ],
            "relationships": ["CONTAINS", "CALLS"],
            "path_length": 2,
        }
    ]

    path_finder = GraphPathFinder(mock_store)
    steps = path_finder.find_relationship("OrderController", "process_payment")

    assert len(steps) == 2
    assert steps[0]["relationship"] == "CONTAINS"
    assert steps[0]["from"]["name"] == "OrderController"
    assert steps[0]["to"]["name"] == "create_order"
    assert steps[1]["relationship"] == "CALLS"
    assert steps[1]["to"]["name"] == "process_payment"
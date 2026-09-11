from unittest.mock import MagicMock
from graph.traversal import GraphTraversal


def test_get_call_chain():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "chain": [
                {"id": "1", "name": "create_order", "file_path": "order.py"},
                {"id": "2", "name": "process_payment", "file_path": "payment.py"},
            ],
            "depth": 1,
        }
    ]

    traversal = GraphTraversal(store=mock_store)
    chains = traversal.get_call_chain("create_order", depth=2)

    assert len(chains) == 1
    assert len(chains[0]["chain"]) == 2
    assert chains[0]["chain"][1]["name"] == "process_payment"
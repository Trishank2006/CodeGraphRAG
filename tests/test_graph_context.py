from unittest.mock import MagicMock
from graph.context import GraphContextExtractor


def test_get_graph_context():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "root_id": "fn:process_payment",
            "root_name": "process_payment",
            "root_type": "Function",
            "root_file": "payment/service.py",
            "connections": [
                {
                    "target_id": "fn:validate",
                    "target_name": "validate",
                    "target_type": "Function",
                    "target_file": "payment/validator.py",
                    "relationship": "CALLS",
                    "distance": 1,
                },
                {
                    "target_id": "class:PaymentGateway",
                    "target_name": "PaymentGateway",
                    "target_type": "Class",
                    "target_file": "payment/gateway.py",
                    "relationship": "IMPORTS",
                    "distance": 1,
                },
            ],
        }
    ]

    extractor = GraphContextExtractor(mock_store)
    ctx = extractor.get_graph_context("process_payment", hops=1)

    assert ctx["name"] == "process_payment"
    assert len(ctx["connections"]) == 2
    assert ctx["connections"][0]["relationship"] == "CALLS"
    assert ctx["connections"][1]["relationship"] == "IMPORTS"
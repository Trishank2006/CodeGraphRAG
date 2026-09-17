from unittest.mock import MagicMock
from graph.grounding import GraphGroundingService


def test_grounding_service_evidence():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "root_id": "fn:login",
            "root_name": "login",
            "root_file": "auth.py",
            "root_start": 10,
            "root_end": 25,
            "connections": [
                {
                    "target_id": "fn:verify_password",
                    "target_name": "verify_password",
                    "target_file": "crypto.py",
                    "target_start": 40,
                    "target_end": 50,
                    "relationship": "CALLS",
                    "distance": 1,
                }
            ],
        }
    ]

    service = GraphGroundingService(mock_store)
    evidence = service.get_evidence("login")

    assert len(evidence) == 1
    item = evidence[0]
    assert item.source_entity == "login"
    assert item.relationship == "CALLS"
    assert item.target_entity == "verify_password"
    assert item.source_file == "auth.py"
    assert item.target_file == "crypto.py"
    assert item.source_start_line == 10
    assert item.target_start_line == 40


def test_mode_aware_context():
    mock_store = MagicMock()
    # Mock caller query response
    mock_store.execute_query.return_value = [
        {"name": "parent_fn", "file_path": "main.py"}
    ]

    service = GraphGroundingService(mock_store)
    res = service.get_mode_aware_context("child_fn", "Who calls child_fn?")
    assert "Callers of child_fn:" in res
    assert "parent_fn" in res
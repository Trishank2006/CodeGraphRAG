from unittest.mock import MagicMock
from graph.repository import GraphRepositoryManager


def test_repository_stats():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {"node_count": 15, "rel_count": 30, "labels": ["Function", "Class"]}
    ]

    manager = GraphRepositoryManager(mock_store)
    stats = manager.get_repository_stats("sample_repo")

    assert stats["repository"] == "sample_repo"
    assert stats["node_count"] == 15
    assert stats["rel_count"] == 30
    assert "Function" in stats["labels"]


def test_repository_delete():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [{"deleted_count": 12}]

    manager = GraphRepositoryManager(mock_store)
    deleted = manager.delete_repository("sample_repo")

    assert deleted == 12
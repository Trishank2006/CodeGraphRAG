from unittest.mock import MagicMock
from graph.health import check_graph_health


def test_graph_health_healthy():
    mock_store = MagicMock()
    mock_store.database = "neo4j"
    mock_store.execute_query.side_effect = [
        [{"ping": 1}],
        [{"node_count": 42, "rel_count": 84}],
    ]

    res = check_graph_health(mock_store)
    assert res["status"] == "healthy"
    assert res["connected"] is True
    assert res["database"] == "neo4j"
    assert res["node_count"] == 42
    assert res["relationship_count"] == 84


def test_graph_health_connection_failure():
    mock_store = MagicMock()
    mock_store.execute_query.side_effect = Exception("Connection refused")

    res = check_graph_health(mock_store)
    assert res["status"] == "unhealthy"
    assert res["connected"] is False
    assert "Connection refused" in res["error"]
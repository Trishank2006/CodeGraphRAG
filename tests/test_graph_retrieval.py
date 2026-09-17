from unittest.mock import MagicMock
from graph.retrieval import GraphRetriever, graph_search
from retrieval.models import RetrievalResult


def test_get_callers_returns_retrieval_result():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "id": "file:service/order.py:function:create_order",
            "name": "create_order",
            "label": "Function",
            "file_path": "service/order.py",
            "language": "python",
            "start_line": 10,
            "end_line": 20,
            "repository": "repo",
        }
    ]

    retriever = GraphRetriever(mock_store)
    results = retriever.get_callers("process_payment")

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, RetrievalResult)
    assert res.source == "graph"
    assert res.symbol == "create_order"
    assert res.score == 1.0
    assert res.file_path == "service/order.py"


def test_get_callees_returns_retrieval_result():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "id": "file:service/payment.py:function:process_payment",
            "name": "process_payment",
            "label": "Function",
            "file_path": "service/payment.py",
            "language": "python",
            "start_line": 30,
            "end_line": 45,
            "repository": "repo",
        }
    ]

    retriever = GraphRetriever(mock_store)
    results = retriever.get_callees("create_order")

    assert len(results) == 1
    assert results[0].symbol == "process_payment"
    assert results[0].source == "graph"


def test_get_dependencies_returns_retrieval_result():
    mock_store = MagicMock()
    mock_store.execute_query.return_value = [
        {
            "id": "module:database",
            "name": "database",
            "label": "File",
            "file_path": "db/database.py",
            "language": "python",
            "start_line": 1,
            "end_line": 50,
            "repository": "repo",
        }
    ]

    retriever = GraphRetriever(mock_store)
    results = retriever.get_dependencies("service/order.py")

    assert len(results) == 1
    assert results[0].symbol == "database"
    assert results[0].score == 0.8
    assert results[0].source == "graph"


def test_get_related_code_hops_control():
    mock_store = MagicMock()
    # Mock return with distance metadata
    mock_store.execute_query.return_value = [
        {
            "id": "1",
            "name": "validator",
            "file_path": "validator.py",
            "language": "python",
            "start_line": 5,
            "end_line": 15,
            "repository": "repo",
            "distance": 1,
        },
        {
            "id": "2",
            "name": "database",
            "file_path": "db.py",
            "language": "python",
            "start_line": 1,
            "end_line": 30,
            "repository": "repo",
            "distance": 2,
        },
    ]

    retriever = GraphRetriever(mock_store)
    results = retriever.get_related_code("process_payment", hops=2)

    assert len(results) == 2
    # Distance 1 scores 1.0, distance 2 scores 0.7
    assert results[0].score == 1.0
    assert results[1].score == 0.7
    assert all(r.source == "graph" for r in results)


def test_graph_search_top_level_api():
    mock_store = MagicMock()
    # find_entities returns seed
    mock_store.execute_query.side_effect = [
        [
            {
                "id": "entity:1",
                "name": "PaymentService",
                "file_path": "pay.py",
                "language": "python",
                "start_line": 1,
                "end_line": 40,
                "repository": "repo",
            }
        ],
        [],  # neighbors return empty
    ]

    results = graph_search("PaymentService", store=mock_store, top_k=5)

    assert len(results) == 1
    assert results[0].symbol == "PaymentService"
    assert isinstance(results[0], RetrievalResult)
    assert results[0].source == "graph"
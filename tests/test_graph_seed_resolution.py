from unittest.mock import MagicMock
from graph.retrieval import GraphRetriever


def test_seed_resolution_and_filtering():
    mock_store = MagicMock()
    mock_store.execute_query.side_effect = [
        [
            {
                "id": "fn:authenticate",
                "name": "authenticate_user",
                "label": "Function",
                "file_path": "auth/service.py",
                "language": "python",
                "start_line": 10,
                "end_line": 35,
                "repository": "repo",
            }
        ],
        [],  # neighbors return empty
    ]

    retriever = GraphRetriever(mock_store)
    results = retriever.search(
        query="user authentication",
        entity_type="Function",
        language="python",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].symbol == "authenticate_user"
    assert results[0].language == "python"
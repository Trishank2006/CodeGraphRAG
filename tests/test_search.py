from retrieval.models import RetrievalResult
from retrieval.search import search_code


class FakeEmbedder:
    def embed_query(self, query):
        assert query == "payment processing"
        return [1.0, 0.0, 0.0]


class FakeStore:
    def search(
        self,
        query_vector,
        top_k,
        language=None,
        file_path=None,
        path_prefix=None,
    ):
        assert query_vector == [1.0, 0.0, 0.0]
        assert top_k == 5
        assert language == "python"
        assert file_path is None
        assert path_prefix == "src/payment/"

        return [
            {
                "score": 0.99,
                "payload": {
                    "chunk_id": "chunk-1",
                    "repository": "test-repository",
                    "file_path": "src/payment/service.py",
                    "language": "python",
                    "symbol": "process_payment",
                    "start_line": 1,
                    "end_line": 5,
                    "content": "def process_payment():\n    pass",
                },
            }
        ]


def test_search_code():
    results = search_code(
        query="payment processing",
        top_k=5,
        language="python",
        path_prefix="src/payment/",
        embedder=FakeEmbedder(),
        store=FakeStore(),
    )

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)

    assert results[0].id == "chunk-1"
    assert results[0].source == "vector"
    assert results[0].score == 0.99
    assert results[0].symbol == "process_payment"
    assert results[0].file_path == "src/payment/service.py"
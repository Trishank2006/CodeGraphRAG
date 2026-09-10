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
    ):
        assert query_vector == [1.0, 0.0, 0.0]
        assert top_k == 5
        assert language == "python"
        assert file_path == "src/payment.py"

        return [
            {
                "score": 0.99,
                "payload": {
                    "chunk_id": "chunk-1",
                    "symbol": "process_payment",
                },
            }
        ]


def test_search_code():
    results = search_code(
        query="payment processing",
        top_k=5,
        language="python",
        file_path="src/payment.py",
        embedder=FakeEmbedder(),
        store=FakeStore(),
    )

    assert len(results) == 1
    assert results[0]["payload"]["symbol"] == "process_payment"
from retrieval.search import search_code
from vector_store.indexer import index_chunks
from vector_store.qdrant_store import QdrantStore


class FakeEmbedder:
    def embed(self, texts):
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


def test_end_to_end_search(tmp_path):
    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        vector_size=3,
    )

    embedder = FakeEmbedder()

    chunks = [
        {
            "chunk_id": "chunk-1",
            "repository": "test-repository",
            "file_path": "src/payment.py",
            "language": "python",
            "symbol": "process_payment",
            "start_line": 1,
            "end_line": 2,
            "content": "def process_payment():\n    pass",
        },
        {
            "chunk_id": "chunk-2",
            "repository": "test-repository",
            "file_path": "src/user.py",
            "language": "python",
            "symbol": "create_user",
            "start_line": 1,
            "end_line": 2,
            "content": "def create_user():\n    pass",
        },
    ]

    index_chunks(
        chunks=chunks,
        embedder=embedder,
        store=store,
    )

    results = search_code(
        query="payment processing",
        top_k=1,
        embedder=embedder,
        store=store,
    )

    assert len(results) == 1
    assert results[0]["payload"]["chunk_id"] == "chunk-1"
    assert results[0]["payload"]["symbol"] == "process_payment"
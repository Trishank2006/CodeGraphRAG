from embeddings.embedder import CodeEmbedder
from retrieval.search import search_code
from vector_store.indexer import index_chunks
from vector_store.qdrant_store import QdrantStore


def test_real_embedding_and_qdrant_pipeline(tmp_path):
    chunks = [
        {
            "chunk_id": "real-payment",
            "repository": "demo-repository",
            "file_path": "src/payment.py",
            "language": "python",
            "symbol": "process_payment",
            "start_line": 1,
            "end_line": 5,
            "content": """
def process_payment(amount):
    validate_payment(amount)
    charge_card(amount)
    return True
""".strip(),
        },
        {
            "chunk_id": "real-user",
            "repository": "demo-repository",
            "file_path": "src/user.py",
            "language": "python",
            "symbol": "create_user",
            "start_line": 1,
            "end_line": 5,
            "content": """
def create_user(name, email):
    user = User(name, email)
    save_user(user)
    return user
""".strip(),
        },
    ]

    embedder = CodeEmbedder()

    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="real_test_collection",
        vector_size=384,
    )

    index_chunks(
        chunks=chunks,
        embedder=embedder,
        store=store,
    )

    results = search_code(
        query="How is payment processed?",
        top_k=1,
        embedder=embedder,
        store=store,
    )

    assert len(results) == 1
    assert results[0]["payload"]["symbol"] == "process_payment"
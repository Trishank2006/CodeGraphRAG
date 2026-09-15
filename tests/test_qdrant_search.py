from vector_store.qdrant_store import QdrantStore


def create_test_store(tmp_path):
    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        vector_size=3,
    )

    store.create_collection()

    chunks = [
        {
            "chunk_id": "chunk-1",
            "repository": "test-repository",
            "file_path": "src/payment.py",
            "language": "python",
            "symbol": "process_payment",
            "start_line": 1,
            "end_line": 5,
            "content": "def process_payment():\n    pass",
        },
        {
            "chunk_id": "chunk-2",
            "repository": "test-repository",
            "file_path": "src/user.py",
            "language": "python",
            "symbol": "create_user",
            "start_line": 1,
            "end_line": 4,
            "content": "def create_user():\n    pass",
        },
        {
            "chunk_id": "chunk-3",
            "repository": "test-repository",
            "file_path": "src/payment.js",
            "language": "javascript",
            "symbol": "processPayment",
            "start_line": 1,
            "end_line": 5,
            "content": "function processPayment() {}",
        },
        {
            "chunk_id": "chunk-4",
            "repository": "test-repository",
            "file_path": "src/payment/service.py",
            "language": "python",
            "symbol": "validate_payment",
            "start_line": 1,
            "end_line": 6,
            "content": "def validate_payment():\n    pass",
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.9, 0.1, 0.0],
    ]

    store.upsert_chunks(chunks, embeddings)

    return store


def test_search_returns_most_similar_chunks(tmp_path):
    store = create_test_store(tmp_path)

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["payload"]["chunk_id"] == "chunk-1"
    assert results[0]["payload"]["symbol"] == "process_payment"


def test_search_filters_by_language(tmp_path):
    store = create_test_store(tmp_path)

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=10,
        language="javascript",
    )

    assert len(results) == 1
    assert results[0]["payload"]["chunk_id"] == "chunk-3"
    assert results[0]["payload"]["language"] == "javascript"


def test_search_filters_by_file_path(tmp_path):
    store = create_test_store(tmp_path)

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=10,
        file_path="src/user.py",
    )

    assert len(results) == 1
    assert results[0]["payload"]["chunk_id"] == "chunk-2"
    assert results[0]["payload"]["file_path"] == "src/user.py"


def test_search_filters_by_path_prefix(tmp_path):
    store = create_test_store(tmp_path)

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=10,
        path_prefix="src/payment/",
    )

    assert len(results) == 1
    assert results[0]["payload"]["chunk_id"] == "chunk-4"
    assert results[0]["payload"]["file_path"] == "src/payment/service.py"
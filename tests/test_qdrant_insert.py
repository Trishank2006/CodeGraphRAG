from vector_store.qdrant_store import QdrantStore, point_id_for


def _chunk(chunk_id, repository="test-repository"):
    return {
        "chunk_id": chunk_id,
        "repository": repository,
        "file_path": f"src/{chunk_id}.py",
        "language": "python",
        "symbol": chunk_id,
        "start_line": 1,
        "end_line": 2,
        "content": f"def {chunk_id}():\n    pass",
    }


def test_upsert_chunks(tmp_path):
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
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.upsert_chunks(chunks, embeddings)

    point_ids = [point_id_for(chunk) for chunk in chunks]

    result = store.client.retrieve(
        collection_name="test_collection",
        ids=point_ids,
        with_payload=True,
        with_vectors=True,
    )

    assert len(result) == 2
    assert result[0].payload["chunk_id"] == "chunk-1"
    assert result[0].payload["symbol"] == "process_payment"
    assert result[1].payload["chunk_id"] == "chunk-2"
    assert result[1].payload["symbol"] == "create_user"

    store.close()


def test_replace_repository_prunes_stale_points_and_isolates_repositories(tmp_path):
    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        vector_size=3,
    )
    store.create_collection()

    # Same chunk ID "a" in two repositories must not collide.
    store.upsert_chunks(
        [_chunk("a"), _chunk("b"), _chunk("a", repository="other")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    store.replace_repository(
        "test-repository",
        [_chunk("a"), _chunk("c")],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )

    remaining = {
        chunk["chunk_id"]
        for chunk in store.load_repository_chunks("test-repository")
    }
    assert remaining == {"a", "c"}  # "b" was stale and is gone
    assert store.count_repository("other") == 1

    store.close()
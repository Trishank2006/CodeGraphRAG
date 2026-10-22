from qdrant_client import QdrantClient

from vector_store.qdrant_store import QdrantStore


def test_create_collection(tmp_path):
    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        vector_size=3,
    )

    store.create_collection()

    collections = store.client.get_collections()

    names = {
        collection.name
        for collection in collections.collections
    }

    assert "test_collection" in names


def test_existing_collection_is_not_recreated(tmp_path):
    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        vector_size=3,
    )

    store.create_collection()

    first_collections = store.client.get_collections()

    store.create_collection()

    second_collections = store.client.get_collections()

    assert len(first_collections.collections) == len(
        second_collections.collections
    )


def test_delete_repository_preserves_other_repositories(tmp_path):
    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        vector_size=3,
    )

    try:
        store.create_collection()

        store.upsert_chunks(
            chunks=[
                {
                    "chunk_id": "demo:main.py:1:5",
                    "repository": "demo",
                    "file_path": "main.py",
                },
                {
                    "chunk_id": "other:app.py:1:5",
                    "repository": "other",
                    "file_path": "app.py",
                },
            ],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        store.delete_repository("demo")

        remaining = store.client.scroll(
            collection_name="test_collection",
            limit=10,
            with_payload=True,
            with_vectors=False,
        )[0]

        remaining_repositories = {
            point.payload["repository"]
            for point in remaining
        }

        assert remaining_repositories == {"other"}
    finally:
        store.close()

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
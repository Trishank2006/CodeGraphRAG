from vector_store.indexer import index_chunks


class FakeEmbedder:
    def embed(self, texts):
        assert texts == [
            "def process_payment():\n    pass",
            "def create_user():\n    pass",
        ]

        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]


class FakeStore:
    def __init__(self):
        self.collection_created = False
        self.received_chunks = None
        self.received_embeddings = None

    def create_collection(self):
        self.collection_created = True

    def upsert_chunks(self, chunks, embeddings):
        self.received_chunks = chunks
        self.received_embeddings = embeddings


def test_index_chunks():
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

    store = FakeStore()

    index_chunks(
        chunks=chunks,
        embedder=FakeEmbedder(),
        store=store,
    )

    assert store.collection_created is True
    assert store.received_chunks == chunks
    assert store.received_embeddings == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
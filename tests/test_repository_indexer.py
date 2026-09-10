from vector_store.repository_indexer import index_repository_files


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


def test_index_repository_files():
    files = [
        {
            "repository": "test-repository",
            "file_path": "src/payment.py",
            "language": "python",
            "content": "def process_payment():\n    pass",
        },
        {
            "repository": "test-repository",
            "file_path": "src/user.py",
            "language": "python",
            "content": "def create_user():\n    pass",
        },
    ]

    entities_by_file = {
        "src/payment.py": [
            {
                "name": "process_payment",
                "type": "function",
                "start_line": 1,
                "end_line": 2,
            }
        ],
        "src/user.py": [
            {
                "name": "create_user",
                "type": "function",
                "start_line": 1,
                "end_line": 2,
            }
        ],
    }

    store = FakeStore()

    index_repository_files(
        files=files,
        entities_by_file=entities_by_file,
        embedder=FakeEmbedder(),
        store=store,
    )

    assert store.collection_created is True

    assert len(store.received_chunks) == 2

    assert store.received_chunks[0]["symbol"] == "process_payment"
    assert store.received_chunks[1]["symbol"] == "create_user"

    assert store.received_embeddings == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
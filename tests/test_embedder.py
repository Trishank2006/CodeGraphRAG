import numpy as np

from embeddings.embedder import CodeEmbedder


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([0.1, 0.2, 0.3])

        return np.array([
            [0.1, 0.2, 0.3]
            for _ in texts
        ])


def test_embed():
    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.model = FakeModel()

    embeddings = embedder.embed(
        [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b",
        ]
    )

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.1, 0.2, 0.3]


def test_embed_query():
    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.model = FakeModel()

    embedding = embedder.embed_query(
        "How does addition work?"
    )

    assert embedding == [0.1, 0.2, 0.3]
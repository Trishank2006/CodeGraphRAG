from embeddings.embedder import CodeEmbedder
from vector_store.qdrant_store import QdrantStore


def index_chunks(
    chunks: list[dict],
    embedder: CodeEmbedder | None = None,
    store: QdrantStore | None = None,
) -> None:
    """
    Embed code chunks and store them in Qdrant.
    """

    if not chunks:
        return

    if embedder is None:
        embedder = CodeEmbedder()

    if store is None:
        store = QdrantStore()

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = embedder.embed(texts)

    store.create_collection()

    store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )
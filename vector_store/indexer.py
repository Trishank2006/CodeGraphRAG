from embeddings.embedder import CodeEmbedder
from vector_store.qdrant_store import QdrantStore


def embed_chunks(
    chunks: list[dict],
    embedder: CodeEmbedder | None = None,
) -> list[list[float]]:
    """Embed chunk contents without touching any store."""
    if not chunks:
        return []

    embedder = embedder or CodeEmbedder()
    return embedder.embed([chunk["content"] for chunk in chunks])


def write_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    store: QdrantStore | None = None,
) -> None:
    """Store already-embedded chunks in Qdrant."""
    if not chunks:
        return

    store = store or QdrantStore()
    store.create_collection()
    store.upsert_chunks(chunks=chunks, embeddings=embeddings)


def index_chunks(
    chunks: list[dict],
    embedder: CodeEmbedder | None = None,
    store: QdrantStore | None = None,
) -> None:
    """Embed code chunks and store them in Qdrant (kept for existing callers)."""
    if not chunks:
        return

    write_chunks(chunks, embed_chunks(chunks, embedder), store)
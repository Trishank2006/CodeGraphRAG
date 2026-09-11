from embeddings.embedder import CodeEmbedder
from vector_store.qdrant_store import QdrantStore


def search_code(
    query: str,
    top_k: int = 20,
    language: str | None = None,
    file_path: str | None = None,
    embedder: CodeEmbedder | None = None,
    store: QdrantStore | None = None,
) -> list[dict]:
    """
    Search the indexed codebase using a natural-language query.
    """

    if embedder is None:
        embedder = CodeEmbedder()

    if store is None:
        store = QdrantStore()

    query_vector = embedder.embed_query(query)

    return store.search(
        query_vector=query_vector,
        top_k=top_k,
        language=language,
        file_path=file_path,
    )
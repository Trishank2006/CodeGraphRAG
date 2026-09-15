from embeddings.embedder import CodeEmbedder
from retrieval.models import RetrievalResult
from vector_store.qdrant_store import QdrantStore


def search_code(
    query: str,
    top_k: int = 20,
    language: str | None = None,
    file_path: str | None = None,
    path_prefix: str | None = None,
    embedder: CodeEmbedder | None = None,
    store: QdrantStore | None = None,
) -> list[RetrievalResult]:
    """
    Perform dense semantic retrieval over indexed code.

    Supports:
    - natural-language queries
    - top-k retrieval
    - language filtering
    - exact file-path filtering
    - path-prefix filtering
    """

    if embedder is None:
        embedder = CodeEmbedder()

    if store is None:
        store = QdrantStore()

    query_vector = embedder.embed_query(query)

    results = store.search(
        query_vector=query_vector,
        top_k=top_k,
        language=language,
        file_path=file_path,
        path_prefix=path_prefix,
    )

    retrieval_results = []

    for result in results:
        payload = result["payload"]

        retrieval_results.append(
            RetrievalResult(
                id=payload["chunk_id"],
                source="vector",
                score=result["score"],
                repository=payload["repository"],
                file_path=payload["file_path"],
                language=payload["language"],
                symbol=payload["symbol"],
                start_line=payload["start_line"],
                end_line=payload["end_line"],
                content=payload["content"],
            )
        )

    return retrieval_results
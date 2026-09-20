from chunking.ast_chunker import chunk_entities
from chunking.code_chunker import create_chunk
from embeddings.embedder import CodeEmbedder
from vector_store.indexer import index_chunks
from vector_store.qdrant_store import QdrantStore


def index_repository_files(
    files: list[dict],
    entities_by_file: dict[str, list[dict]],
    embedder: CodeEmbedder | None = None,
    store: QdrantStore | None = None,
) -> None:
    """
    Convert parsed repository files into chunks and index them.

    files:
        Output from the repository ingestion pipeline.

    entities_by_file:
        Mapping from file path to AST entities produced by Person 2.
    """
    chunks = build_repository_chunks(files, entities_by_file)

    index_chunks(
        chunks=chunks,
        embedder=embedder,
        store=store,
    )


def build_repository_chunks(
    files: list[dict],
    entities_by_file: dict[str, list[dict]],
) -> list[dict]:
    """Build reusable vector/BM25 chunks from parsed repository files."""
    chunks: list[dict] = []

    for file_data in files:
        file_path = file_data["file_path"]
        entities = entities_by_file.get(file_path, [])
        content = file_data.get("content", "")

        file_chunks = []
        if entities:
            file_chunks = chunk_entities(
                repository=file_data["repository"],
                file_path=file_path,
                language=file_data.get("language", "unknown"),
                content=content,
                entities=entities,
            )

        # Fallback: if no AST entities are detected, chunk the whole file
        if not file_chunks and content.strip():
            lines = content.splitlines()
            fallback_chunk = create_chunk(
                repository=file_data["repository"],
                file_path=file_path,
                language=file_data.get("language", "unknown"),
                symbol=file_path,
                start_line=1,
                end_line=max(len(lines), 1),
                content=content,
            )
            file_chunks.append(fallback_chunk)

        chunks.extend(
            chunk.__dict__
            for chunk in file_chunks
        )

    return chunks
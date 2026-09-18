from chunking.ast_chunker import chunk_entities
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

        file_chunks = chunk_entities(
            repository=file_data["repository"],
            file_path=file_path,
            language=file_data["language"],
            content=file_data["content"],
            entities=entities,
        )

        chunks.extend(
            chunk.__dict__
            for chunk in file_chunks
        )

    return chunks

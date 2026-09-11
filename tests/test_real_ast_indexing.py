from parser.core_parser import parse_source_code
from vector_store.repository_indexer import index_repository_files
from vector_store.qdrant_store import QdrantStore
from embeddings.embedder import CodeEmbedder


def test_real_ast_to_qdrant(tmp_path):
    file_path = "tests/fixtures/sample.py"

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    parsed = parse_source_code(
        file_path=file_path,
        content=content,
        language="python",
    )

    assert len(parsed.entities) == 3

    files = [
        {
            "repository": "CodeGraphRAG-test",
            "file_path": file_path,
            "language": "python",
            "content": content,
        }
    ]

    entities_by_file = {
        file_path: parsed.entities
    }

    embedder = CodeEmbedder()

    store = QdrantStore(
        path=str(tmp_path / "qdrant"),
        collection_name="real_ast_test",
        vector_size=384,
    )

    index_repository_files(
        files=files,
        entities_by_file=entities_by_file,
        embedder=embedder,
        store=store,
    )

    results = store.search(
        query_vector=embedder.embed_query(
            "payment processing"
        ),
        top_k=3,
    )

    assert len(results) > 0

    symbols = [
        result["payload"]["symbol"]
        for result in results
    ]

    assert "process_payment" in symbols
import os
import shutil
import stat

from embeddings.embedder import CodeEmbedder
from graph.builder import GraphBuilder
from ingestion.main import ingest_repository
from parser.core_parser import parse_source_code
from vector_store.repository_indexer import index_repository_files


REPOSITORY_URL = "https://github.com/Akki-24/meeting_summarizer.git"
DESTINATION = "./temp_test_repo"


def remove_readonly(func, path, exc_info):
    """Clear the readonly bit and reattempt the removal."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    # Clean up the previous temporary repository.
    if os.path.exists(DESTINATION):
        shutil.rmtree(
            DESTINATION,
            onexc=remove_readonly,
        )

    print("Step 1: Repository ingestion")

    files_data = ingest_repository(
        REPOSITORY_URL,
        DESTINATION,
    )

    print(
        f"Ingestion complete. "
        f"Discovered {len(files_data)} supported files."
    )

    print("\nStep 2: AST parsing")

    entities_by_file = {}
    parsed_files_list = []
    total_entities = 0

    for file_record in files_data:
        parsed = parse_source_code(
            file_path=file_record["file_path"],
            content=file_record["content"],
            language=file_record["language"],
        )

        parsed_files_list.append(parsed)
        entities_by_file[file_record["file_path"]] = parsed.entities

        if parsed.entities:
            print(f"\nFile: {parsed.file_path}")

            for entity in parsed.entities:
                print(
                    f"  [{entity.type}] "
                    f"{entity.name} "
                    f"(Lines "
                    f"{entity.start_line + 1}-"
                    f"{entity.end_line + 1})"
                )

            total_entities += len(parsed.entities)

    print(
        f"\nAST parsing complete. "
        f"Extracted {total_entities} entities."
    )

    print("\nStep 2.5: Building Code Knowledge Graph (Phase 2 - Person 2)")

    builder = GraphBuilder(repository_name="meeting_summarizer")
    graph_nodes, graph_edges = builder.build_from_parsed_files(parsed_files_list)

    print(
        f"Graph construction complete. "
        f"Generated {len(graph_nodes)} nodes and {len(graph_edges)} edges."
    )

    # Breakdown by edge types
    edge_counts = {}
    for edge in graph_edges:
        edge_counts[edge.type] = edge_counts.get(edge.type, 0) + 1

    for rel_type, count in edge_counts.items():
        print(f"  - {rel_type}: {count} relationships")

    print("\nStep 3: Semantic indexing (Phase 2 - Person 1)")

    embedder = CodeEmbedder()

    index_repository_files(
        files=files_data,
        entities_by_file=entities_by_file,
        embedder=embedder,
    )

    print("Semantic indexing complete.")

    print("\nStep 4: Semantic search")

    from retrieval.search import search_code
    from vector_store.qdrant_store import QdrantStore

    store = QdrantStore()

    results = search_code(
        query="How is authentication implemented?",
        top_k=5,
        embedder=embedder,
        store=store,
    )

    for index, result in enumerate(results, start=1):
        payload = result["payload"]

        print(f"\n{index}. {payload['symbol']}")
        print(f"   File: {payload['file_path']}")
        print(f"   Language: {payload['language']}")
        print(
            f"   Lines: "
            f"{payload['start_line']}-"
            f"{payload['end_line']}"
        )
        print(f"   Score: {result['score']:.4f}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
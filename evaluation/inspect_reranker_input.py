from application.service import CodeGraphRAGService
from evaluation.service_retriever import ServiceRetriever
from retrieval.config import RetrievalConfig

service = CodeGraphRAGService()

try:
    service._is_indexed = True

    config = RetrievalConfig(
        dense_top_k=20,
        bm25_top_k=20,
        graph_top_k=20,
        rrf_top_k=30,
        rerank_top_k=5,
    )

    adapter = ServiceRetriever(service, config=config)
    pipeline = adapter._hybrid_pipeline(
        include_graph=True,
        include_reranker=True,
    )

    queries = [
        "Where is BM25 retrieval implemented?",
        "Where is the CLI entry point defined?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print("QUERY:", query)

        candidates = pipeline.hybrid.search(
            query=query,
            dense_top_k=config.dense_top_k,
            bm25_top_k=config.bm25_top_k,
            graph_top_k=config.graph_top_k,
            rrf_top_k=config.rrf_top_k,
        )

        for rank, result in enumerate(candidates, 1):
            if result.file_path in {"retrieval/bm25.py", "cli.py"} or rank <= 3:
                print(f"\nCandidate {rank}")
                print("File:", result.file_path)
                print("Symbol:", result.symbol)
                print("Lines:", result.start_line, "-", result.end_line)
                print("Content length:", len(result.content))
                print("Content preview:")
                print(result.content[:1200])

finally:
    service.close()

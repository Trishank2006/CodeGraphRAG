from __future__ import annotations


from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import statistics
import time
from pathlib import Path

from application.service import CodeGraphRAGService
from evaluation.latency import calculate_latency_stats, latency_stats_to_dict
from retrieval.pipeline import HybridRetrievalPipeline
from graph.retrieval import GraphRetriever
from retrieval.search import search_code


QUERIES = [
    "How does repository indexing work?",
    "Where are embeddings generated?",
    "How does hybrid retrieval combine results?",
    "How are graph relationships stored?",
    "How does BM25 search work?",
]

ITERATIONS = 5


def main() -> None:
    service = CodeGraphRAGService()
    try:
        # Reuse the isolated index created by the baseline benchmark.
        service._is_indexed = True

        graph_retriever = GraphRetriever(service.graph_store)

        def dense(query: str, top_k: int):
            return search_code(
                query=query,
                top_k=top_k,
                embedder=service.embedder,
                store=service.vector_store,
            )

        pipeline = HybridRetrievalPipeline(
            dense_search=dense,
            bm25_retriever=service.bm25_retriever,
            graph_search=lambda query, top_k: graph_retriever.search(
                query, top_k=top_k
            ),
            reranker=service.reranker,
        )

        stage_durations: dict[str, list[float]] = {
            "dense": [],
            "bm25": [],
            "graph": [],
            "fusion": [],
            "reranking": [],
            "end_to_end": [],
        }

        # Warm up model loading before measuring.
        for query in QUERIES:
            pipeline.search(query)

        for query in QUERIES:
            for _ in range(ITERATIONS):
                dense_start = time.perf_counter()
                dense_results = dense(query, 20)
                stage_durations["dense"].append(
                    (time.perf_counter() - dense_start) * 1000
                )

                bm25_start = time.perf_counter()
                bm25_results = service.bm25_retriever.search_bm25(
                    query, top_k=20
                )
                stage_durations["bm25"].append(
                    (time.perf_counter() - bm25_start) * 1000
                )

                graph_start = time.perf_counter()
                graph_results = graph_retriever.search(query, top_k=20)
                stage_durations["graph"].append(
                    (time.perf_counter() - graph_start) * 1000
                )

                fusion_start = time.perf_counter()
                fused = pipeline.hybrid.search(
                    query=query,
                    dense_top_k=20,
                    bm25_top_k=20,
                    graph_top_k=20,
                    rrf_top_k=30,
                )
                stage_durations["fusion"].append(
                    (time.perf_counter() - fusion_start) * 1000
                )

                rerank_start = time.perf_counter()
                pipeline.reranker.rerank(query, fused, top_k=10)
                stage_durations["reranking"].append(
                    (time.perf_counter() - rerank_start) * 1000
                )

                total_start = time.perf_counter()
                pipeline.search(query)
                stage_durations["end_to_end"].append(
                    (time.perf_counter() - total_start) * 1000
                )

        report = {
            "queries": len(QUERIES),
            "iterations_per_query": ITERATIONS,
            "measurements": {
                stage: latency_stats_to_dict(
                    calculate_latency_stats(values)
                )
                for stage, values in stage_durations.items()
            },
            "note": (
                "Stage timings are measured separately. Fusion includes "
                "the retrieval calls made by HybridRetriever; compare "
                "with component timings carefully and do not sum them."
            ),
        }

        output = Path("evaluation/results/retrieval_profile.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n=== Retrieval Component Profile ===")
        for stage, stats in report["measurements"].items():
            print(
                f"{stage:15} "
                f"p50={stats['p50_ms']:.2f} ms  "
                f"p95={stats['p95_ms']:.2f} ms"
            )
        print(f"\nSaved: {output.resolve()}")

    finally:
        service.close()


if __name__ == "__main__":
    main()


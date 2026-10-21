from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.service import CodeGraphRAGService
from graph.retrieval import GraphRetriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.search import search_code

QUERIES = [
    "How does repository indexing work?",
    "Where are embeddings generated?",
    "How does hybrid retrieval combine results?",
    "How are graph relationships stored?",
    "How does BM25 search work?",
]

CANDIDATE_COUNTS = (10, 20, 30)
ITERATIONS = 5
OUTPUT_PATH = Path("evaluation/results/rerank_candidates.json")


def percentile(values: list[float], p: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * weight


def build_candidates(
    service: CodeGraphRAGService,
    graph: GraphRetriever,
    query: str,
) -> list:
    """Build the shared hybrid candidate ranking before reranking."""
    dense_results = search_code(
        query=query,
        top_k=20,
        embedder=service.embedder,
        store=service.vector_store,
    )

    bm25_results = service.bm25_retriever.search_bm25(
        query,
        top_k=20,
    )

    graph_results = graph.search(query, top_k=20)

    return reciprocal_rank_fusion(
        [dense_results, bm25_results, graph_results],
        k=60,
        top_k=max(CANDIDATE_COUNTS),
    )


def main() -> None:
    service = CodeGraphRAGService()

    try:
        # Use the isolated benchmark stores.
        service._is_indexed = True
        graph = GraphRetriever(service.graph_store)

        # Build candidates once per query so all configurations use
        # the same retrieval results and ranking.
        query_candidates = {
            query: build_candidates(service, graph, query)
            for query in QUERIES
        }

        # Warm up the reranker with actual, non-empty candidate lists.
        # This keeps model initialization out of the timed measurements.
        for query, candidates in query_candidates.items():
            service.reranker.rerank(
                query,
                candidates[:30],
                top_k=10,
            )

        report = {
            "queries": len(QUERIES),
            "iterations_per_query": ITERATIONS,
            "experiments": {},
            "note": (
                "Reranking latency only; retrieval candidates are generated "
                "once per query and shared across configurations. "
                "Retrieval quality must be evaluated separately."
            ),
        }

        for candidate_count in CANDIDATE_COUNTS:
            latencies: list[float] = []
            actual_counts: list[int] = []

            for query, all_candidates in query_candidates.items():
                candidates = all_candidates[:candidate_count]

                # Warm up this exact candidate count before measurement.
                service.reranker.rerank(
                    query,
                    candidates,
                    top_k=10,
                )

                for _ in range(ITERATIONS):
                    start = time.perf_counter()

                    service.reranker.rerank(
                        query,
                        candidates,
                        top_k=10,
                    )

                    elapsed_ms = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed_ms)
                    actual_counts.append(len(candidates))

            report["experiments"][str(candidate_count)] = {
                "candidate_count_mean": statistics.mean(actual_counts),
                "latency_ms": {
                    "count": len(latencies),
                    "mean": statistics.mean(latencies),
                    "p50": percentile(latencies, 0.50),
                    "p95": percentile(latencies, 0.95),
                    "min": min(latencies),
                    "max": max(latencies),
                },
            }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(report, indent=2))
        print(f"Saved: {OUTPUT_PATH.resolve()}")

    finally:
        service.close()


if __name__ == "__main__":
    main()
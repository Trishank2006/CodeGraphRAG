from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.service import CodeGraphRAGService
from evaluation.latency import calculate_latency_stats, latency_stats_to_dict
from evaluation.service_retriever import ServiceRetriever
from retrieval.config import RetrievalConfig

QUERIES = [
    "How does repository indexing work?",
    "Where are embeddings generated?",
    "How does hybrid retrieval combine results?",
    "How are graph relationships stored?",
    "How does BM25 search work?",
]

CANDIDATE_COUNTS = (10, 20, 30)
ITERATIONS = 5


def main():
    service = CodeGraphRAGService()
    try:
        service._is_indexed = True
        report = {
            "queries": len(QUERIES),
            "iterations_per_query": ITERATIONS,
            "experiments": {},
        }

        for count in CANDIDATE_COUNTS:
            config = RetrievalConfig(
                dense_top_k=20,
                bm25_top_k=20,
                graph_top_k=20,
                rrf_top_k=count,
                rerank_top_k=5,
            )
            adapter = ServiceRetriever(service, config=config)
            retrieve = adapter.as_retrievers()[
                "Dense + BM25 + Graph + Reranker"
            ]

            # Warm up each query before collecting timings.
            for query in QUERIES:
                retrieve(type("Example", (), {"query": query})())

            timings = []
            for query in QUERIES:
                example = type("Example", (), {"query": query})()
                for _ in range(ITERATIONS):
                    start = time.perf_counter()
                    retrieve(example)
                    timings.append((time.perf_counter() - start) * 1000)

            report["experiments"][str(count)] = latency_stats_to_dict(
                calculate_latency_stats(timings)
            )

        output = Path("evaluation/results/end_to_end_latency.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Saved: {output.resolve()}")

    finally:
        service.close()


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.service import CodeGraphRAGService
from evaluation.dataset import create_default_dataset
from evaluation.retrieval_benchmark import (
    RetrievalBenchmark,
    results_to_dict,
    BenchmarkResult,
)
from evaluation.service_retriever import ServiceRetriever
from retrieval.config import RetrievalConfig


def main():
    service = CodeGraphRAGService()

    try:
        # Use the isolated benchmark stores created earlier.
        service._is_indexed = True

        dataset = create_default_dataset()
        benchmark = RetrievalBenchmark(dataset, top_k=5)

        results = {}

        for candidate_count in (10, 20, 30):
            config = RetrievalConfig(
                dense_top_k=20,
                bm25_top_k=20,
                graph_top_k=20,
                rrf_top_k=candidate_count,
                rerank_top_k=5,
            )

            adapter = ServiceRetriever(service, config=config)
            retrievers = adapter.as_retrievers()
            reranker = retrievers[
                "Dense + BM25 + Graph + Reranker"
            ]

            # Warm up the model using a real query.
            first_example = next(iter(dataset))
            reranker(first_example)

            name = (
                f"Hybrid + Reranker "
                f"(candidates={candidate_count})"
            )

            # Evaluate once and collect overall and grouped metrics.
            detailed = benchmark.evaluate_detailed(
                name=name,
                retriever=reranker,
            )

            overall = detailed["by_query_type"]["overall"]

            results[str(candidate_count)] = {
                "name": name,
                "query_count": overall["query_count"],
                "hit_rate": overall["hit_rate"],
                "precision": overall["precision"],
                "recall": overall["recall"],
                "mrr": overall["mrr"],
                "ndcg": overall["ndcg"],
                "by_query_type": {
                    query_type: metrics
                    for query_type, metrics
                    in detailed["by_query_type"].items()
                    if query_type != "overall"
                },
            }

        report = {
            "dataset_size": len(dataset),
            "evaluation_at_k": 5,
            "candidate_count_comparison": results,
            "note": (
                "Compare retrieval quality and latency together. "
                "The candidate count changes the RRF pool before "
                "reranking. Per-query-type metrics use the existing "
                "ground-truth relevance rules."
            ),
        }

        output = Path(
            "evaluation/results/rerank_quality.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(report, indent=2))
        print(f"Saved: {output.resolve()}")

    finally:
        service.close()


if __name__ == "__main__":
    main()

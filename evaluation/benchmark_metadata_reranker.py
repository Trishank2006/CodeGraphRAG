
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from application.service import CodeGraphRAGService
from evaluation.dataset import EvaluationExample, create_default_dataset
from evaluation.retrieval_benchmark import RetrievalBenchmark
from evaluation.retrieval_metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.service_retriever import ServiceRetriever
from retrieval.config import RetrievalConfig
from retrieval.metadata_reranker import MetadataAwareReranker
from retrieval.models import RetrievalResult


TOP_K = 5


def unique_file_candidates(
    candidates: Sequence[RetrievalResult],
) -> list[RetrievalResult]:
    """Keep the first candidate for each normalized file path."""
    unique: list[RetrievalResult] = []
    seen: set[str] = set()

    for result in candidates:
        path = result.file_path.replace("\\", "/").casefold()

        if path not in seen:
            seen.add(path)
            unique.append(result)

    return unique


def evaluation_results(
    benchmark: RetrievalBenchmark,
    rankings: dict[str, list[RetrievalResult]],
    example: EvaluationExample,
) -> list[RetrievalResult]:
    """Deduplicate results by the benchmark's evaluation identifier."""
    results = rankings[example.query]
    unique: list[RetrievalResult] = []
    seen: set[str] = set()

    for result in results:
        identifier = benchmark._result_identifier(
            result,
            example.query_type,
        )

        if identifier not in seen:
            seen.add(identifier)
            unique.append(result)

    return unique


def result_summary(
    results: Sequence[RetrievalResult],
) -> list[dict]:
    """Create a JSON-friendly summary of ranked results."""
    return [
        {
            "file_path": result.file_path,
            "symbol": result.symbol,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "score": float(result.score),
        }
        for result in results
    ]


def calculate_query_metrics(
    benchmark: RetrievalBenchmark,
    example: EvaluationExample,
    results: Sequence[RetrievalResult],
) -> dict:
    """Calculate ranking metrics for one query."""
    ranking = benchmark._normalise_results(
        results,
        example.query_type,
    )
    relevant = benchmark._relevant_ids(example)

    return {
        "hit_rate": hit_rate_at_k(ranking, relevant, TOP_K),
        "precision": precision_at_k(ranking, relevant, TOP_K),
        "recall": recall_at_k(ranking, relevant, TOP_K),
        "reciprocal_rank": reciprocal_rank(ranking, relevant),
        "ndcg": ndcg_at_k(ranking, relevant, TOP_K),
    }


def validate_metrics(
    original_metrics: dict,
    metadata_metrics: dict,
    per_query_diagnostics: list[dict],
) -> None:
    """Ensure all reported NDCG values are within [0, 1]."""
    for method_name, metrics in (
        ("original", original_metrics),
        ("metadata_aware", metadata_metrics),
    ):
        for group_name, values in metrics["by_query_type"].items():
            score = values["ndcg"]

            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Invalid NDCG for {method_name}/{group_name}: "
                    f"{score}"
                )

    for item in per_query_diagnostics:
        for method_name in (
            "original_metrics",
            "metadata_aware_metrics",
        ):
            score = item[method_name]["ndcg"]

            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Invalid per-query NDCG for {method_name}, "
                    f"query '{item['query']}': {score}"
                )


def main() -> None:
    service = CodeGraphRAGService()

    try:
        # Reuse the existing indexed repository and benchmark setup.
        service._is_indexed = True

        dataset = create_default_dataset()

        config = RetrievalConfig(
            dense_top_k=20,
            bm25_top_k=20,
            graph_top_k=20,
            rrf_top_k=30,
            rerank_top_k=TOP_K,
        )

        adapter = ServiceRetriever(service, config=config)
        pipeline = adapter._hybrid_pipeline(
            include_graph=True,
            include_reranker=True,
        )

        # Warm up the original model.
        first_example = next(iter(dataset))
        pipeline.search(first_example.query)

        # Reuse the same model weights for a fair comparison.
        metadata_reranker = MetadataAwareReranker(
            model_name=pipeline.reranker.model_name,
            model=pipeline.reranker.model,
        )

        candidate_pools: dict[str, list[RetrievalResult]] = {}
        candidate_counts: dict[str, dict[str, int]] = {}

        # Fetch each hybrid candidate pool once.
        # Deduplicate implementation lookups before reranking.
        for example in dataset:
            candidates = pipeline.hybrid.search(
                query=example.query,
                dense_top_k=config.dense_top_k,
                bm25_top_k=config.bm25_top_k,
                graph_top_k=config.graph_top_k,
                rrf_top_k=config.rrf_top_k,
            )

            original_count = len(candidates)

            if example.query_type == "implementation_lookup":
                candidates = unique_file_candidates(candidates)

            candidate_pools[example.query] = candidates
            candidate_counts[example.query] = {
                "before_deduplication": original_count,
                "after_deduplication": len(candidates),
            }

        original_rankings: dict[str, list[RetrievalResult]] = {}
        metadata_rankings: dict[str, list[RetrievalResult]] = {}

        # Both rerankers receive the same candidate list per query.
        for index, example in enumerate(dataset, start=1):
            candidates = candidate_pools[example.query]

            original_rankings[example.query] = (
                pipeline.reranker.rerank(
                    query=example.query,
                    results=candidates,
                    top_k=TOP_K,
                )
            )

            metadata_rankings[example.query] = (
                metadata_reranker.rerank(
                    query=example.query,
                    results=candidates,
                    top_k=TOP_K,
                )
            )

            print(f"[{index}/{len(dataset)}] {example.query}")

        benchmark = RetrievalBenchmark(
            dataset=dataset,
            top_k=TOP_K,
        )

        # Deduplicate final rankings before aggregate evaluation.
        original_metrics = benchmark.evaluate_detailed(
            "original_cross_encoder",
            lambda example: evaluation_results(
                benchmark,
                original_rankings,
                example,
            ),
        )

        metadata_metrics = benchmark.evaluate_detailed(
            "metadata_aware_cross_encoder",
            lambda example: evaluation_results(
                benchmark,
                metadata_rankings,
                example,
            ),
        )

        metric_names = (
            "hit_rate",
            "precision",
            "recall",
            "mrr",
            "ndcg",
        )

        original_overall = original_metrics["by_query_type"]["overall"]
        metadata_overall = metadata_metrics["by_query_type"]["overall"]

        overall_deltas = {
            metric: round(
                metadata_overall[metric] - original_overall[metric],
                6,
            )
            for metric in metric_names
        }

        # Compare per-query metrics using deduplicated rankings.
        per_query_diagnostics = []

        for example in dataset:
            original_results = evaluation_results(
                benchmark,
                original_rankings,
                example,
            )
            metadata_results = evaluation_results(
                benchmark,
                metadata_rankings,
                example,
            )

            original_query_metrics = calculate_query_metrics(
                benchmark,
                example,
                original_results,
            )

            metadata_query_metrics = calculate_query_metrics(
                benchmark,
                example,
                metadata_results,
            )

            query_deltas = {
                metric: round(
                    metadata_query_metrics[metric]
                    - original_query_metrics[metric],
                    6,
                )
                for metric in original_query_metrics
            }

            ndcg_delta = query_deltas["ndcg"]

            if ndcg_delta > 1e-9:
                status = "improved"
            elif ndcg_delta < -1e-9:
                status = "regressed"
            else:
                status = "unchanged"

            per_query_diagnostics.append(
                {
                    "query": example.query,
                    "query_type": example.query_type,
                    "expected_files": list(example.expected_files),
                    "expected_symbols": list(example.expected_symbols),
                    "candidate_counts": candidate_counts[example.query],
                    "status_by_ndcg": status,
                    "original_metrics": original_query_metrics,
                    "metadata_aware_metrics": metadata_query_metrics,
                    "metric_deltas": query_deltas,
                    "original_top_results": result_summary(
                        original_results
                    ),
                    "metadata_aware_top_results": result_summary(
                        metadata_results
                    ),
                }
            )

        improved = [
            item
            for item in per_query_diagnostics
            if item["status_by_ndcg"] == "improved"
        ]

        regressed = [
            item
            for item in per_query_diagnostics
            if item["status_by_ndcg"] == "regressed"
        ]

        unchanged = [
            item
            for item in per_query_diagnostics
            if item["status_by_ndcg"] == "unchanged"
        ]

        report = {
            "candidate_counts": {
                "dense_top_k": config.dense_top_k,
                "bm25_top_k": config.bm25_top_k,
                "graph_top_k": config.graph_top_k,
                "rrf_top_k": config.rrf_top_k,
            },
            "rerank_top_k": TOP_K,
            "query_count": len(dataset),
            "methodology": {
                "candidate_pool": (
                    "Each query's hybrid candidate pool is fetched once "
                    "and shared by both rerankers."
                ),
                "implementation_lookup": (
                    "Candidates are deduplicated by normalized file path "
                    "before reranking. The first occurrence is retained, "
                    "preserving the hybrid retriever's candidate order."
                ),
                "final_rankings": (
                    "Final rankings are deduplicated by the benchmark's "
                    "evaluation identifier before calculating metrics. "
                    "This avoids repeated file paths or symbols inflating "
                    "ranking scores."
                ),
                "other_query_types": (
                    "Candidate pools for non-implementation queries are "
                    "not deduplicated by file before reranking."
                ),
                "comparison": (
                    "Both methods rerank the same candidates. The "
                    "metadata-aware method includes file path, symbol, "
                    "language, and line range with the code content."
                ),
                "per_query_status": (
                    "Improved, regressed, or unchanged is based on the "
                    "difference in per-query NDCG@5."
                ),
                "limitations": (
                    "The evaluation uses the existing dataset and its "
                    "current relevance labels. Some query types have "
                    "few examples, so results are experimental rather "
                    "than conclusive."
                ),
            },
            "original": original_metrics,
            "metadata_aware": metadata_metrics,
            "delta_metadata_minus_original": overall_deltas,
            "per_query_summary": {
                "improved_by_ndcg": len(improved),
                "regressed_by_ndcg": len(regressed),
                "unchanged_by_ndcg": len(unchanged),
            },
            "per_query_diagnostics": per_query_diagnostics,
        }

        # Validate before writing the report.
        validate_metrics(
            original_metrics,
            metadata_metrics,
            per_query_diagnostics,
        )

        output = Path(
            "evaluation/results/metadata_reranker_comparison.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\n=== Original cross-encoder ===")
        print(json.dumps(original_metrics["by_query_type"], indent=2))

        print("\n=== Metadata-aware cross-encoder ===")
        print(json.dumps(metadata_metrics["by_query_type"], indent=2))

        print("\n=== Overall metric deltas ===")
        print(json.dumps(overall_deltas, indent=2))

        print("\n=== Per-query NDCG comparison ===")
        print(json.dumps(report["per_query_summary"], indent=2))

        print("\n=== Queries that regressed ===")
        if regressed:
            for item in regressed:
                print(
                    f"- {item['query']} "
                    f"(NDCG delta: "
                    f"{item['metric_deltas']['ndcg']:+.4f})"
                )
        else:
            print("No queries regressed.")

        print(f"\nSaved full report to: {output}")
        print("All aggregate and per-query NDCG values are valid.")

    finally:
        service.close()


if __name__ == "__main__":
    main()

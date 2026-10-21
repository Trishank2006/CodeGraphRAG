
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from application.service import CodeGraphRAGService
from evaluation.dataset import create_default_dataset
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
from retrieval.models import RetrievalResult




TOP_K = 5
RRF_K = 60

BASE_VARIANTS = (
    "code_only",
    "path_code",
    "path_symbol_code",
)

RRF_WEIGHTS = (0.5, 0.7, 0.8, 0.9, 1.0)

VARIANTS = BASE_VARIANTS + tuple(
    f"weighted_rrf_{int(alpha * 100)}"
    for alpha in RRF_WEIGHTS
)

def load_bm25_from_qdrant(service) -> int:
    """Populate BM25 from existing Qdrant chunk payloads."""
    client = service.vector_store.client
    collection = service.vector_store.collection_name

    chunks = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            required = {
                "chunk_id",
                "repository",
                "file_path",
                "language",
                "symbol",
                "start_line",
                "end_line",
                "content",
            }

            if required.issubset(payload):
                chunks.append(dict(payload))

        if next_offset is None:
            break

        offset = next_offset

    if not chunks:
        raise RuntimeError(
            "No valid chunks found in the existing Qdrant collection."
        )

    service.bm25_retriever.index(chunks)
    return len(chunks)


def format_candidate(
    result: RetrievalResult,
    variant: str,
) -> str:
    """Format candidate text for a controlled ablation."""
    if variant == "code_only":
        return result.content

    if variant == "path_code":
        return (
            f"File path: {result.file_path}\n"
            f"Code:\n{result.content}"
        )

    if variant == "path_symbol_code":
        return (
            f"File path: {result.file_path}\n"
            f"Symbol: {result.symbol}\n"
            f"Code:\n{result.content}"
        )

    raise ValueError(f"Unknown text-format variant: {variant}")


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


def summarize_candidates(
    candidates: Sequence[RetrievalResult],
    expected_files: Sequence[str],
    expected_symbols: Sequence[str],
) -> list[dict]:
    """Record original hybrid ranking and relevance indicators."""
    expected_file_set = {
        path.replace("\\", "/").casefold()
        for path in expected_files
    }
    expected_symbol_set = {
        symbol.casefold()
        for symbol in expected_symbols
    }

    summary = []

    for rank, result in enumerate(candidates, start=1):
        normalized_path = (
            result.file_path.replace("\\", "/").casefold()
        )
        normalized_symbol = (result.symbol or "").casefold()

        summary.append({
            "hybrid_rank": rank,
            "file_path": result.file_path,
            "symbol": result.symbol,
            "hybrid_score": float(result.score),
            "relevant_file": normalized_path in expected_file_set,
            "relevant_symbol": (
                normalized_symbol in expected_symbol_set
            ),
        })

    return summary


def deduplicate_ranking(
    benchmark: RetrievalBenchmark,
    results: Sequence[RetrievalResult],
    query_type: str,
) -> list[RetrievalResult]:
    """Deduplicate using the benchmark's evaluation identifier."""
    unique: list[RetrievalResult] = []
    seen: set[str] = set()

    for result in results:
        identifier = benchmark._result_identifier(
            result,
            query_type,
        )

        if identifier not in seen:
            seen.add(identifier)
            unique.append(result)

    return unique


def rerank_candidates(
    model,
    query: str,
    candidates: Sequence[RetrievalResult],
    variant: str,
    batch_size: int,
) -> list[RetrievalResult]:
    """Rank the complete candidate pool using the selected text format."""
    if not candidates:
        return []

    pairs = [
        (query, format_candidate(candidate, variant))
        for candidate in candidates
    ]

    scores = model.predict(
        pairs,
        batch_size=batch_size,
    )

    if len(scores) != len(candidates):
        raise ValueError(
            f"{variant}: score count does not match candidate count"
        )

    ranked = [
        replace(candidate, score=float(score))
        for candidate, score in zip(candidates, scores)
    ]

    ranked.sort(key=lambda result: (-result.score, result.id))
    return ranked


def reciprocal_rank_fusion(
    hybrid_candidates: Sequence[RetrievalResult],
    cross_encoder_ranking: Sequence[RetrievalResult],
    alpha: float,
    rrf_k: int = RRF_K,
) -> list[RetrievalResult]:
    """
    Blend hybrid and cross-encoder ranks.

    alpha is the cross-encoder contribution:
      0.5 = 50% cross-encoder, 50% hybrid
      0.7 = 70% cross-encoder, 30% hybrid
      0.8 = 80% cross-encoder, 20% hybrid
      0.9 = 90% cross-encoder, 10% hybrid
      1.0 = cross-encoder ranking only
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    if not hybrid_candidates:
        return []

    hybrid_rank = {
        candidate.id: rank
        for rank, candidate in enumerate(
            hybrid_candidates,
            start=1,
        )
    }

    cross_encoder_rank = {
        candidate.id: rank
        for rank, candidate in enumerate(
            cross_encoder_ranking,
            start=1,
        )
    }

    blended = []

    for candidate in hybrid_candidates:
        identifier = candidate.id
        hybrid_position = hybrid_rank[identifier]
        cross_position = cross_encoder_rank.get(identifier)

        fused_score = (
            (1.0 - alpha) / (rrf_k + hybrid_position)
        )

        if cross_position is not None:
            fused_score += alpha / (rrf_k + cross_position)

        blended.append(
            replace(candidate, score=fused_score)
        )

    blended.sort(key=lambda result: (-result.score, result.id))
    return blended


def evaluate_variant(
    benchmark: RetrievalBenchmark,
    dataset,
    rankings: dict[str, list[RetrievalResult]],
) -> dict:
    """Calculate metrics and preserve complete rankings."""
    per_query = []

    for example in dataset:
        results = deduplicate_ranking(
            benchmark,
            rankings[example.query],
            example.query_type,
        )

        identifiers = benchmark._normalise_results(
            results,
            example.query_type,
        )
        relevant = benchmark._relevant_ids(example)

        per_query.append({
            "query": example.query,
            "query_type": example.query_type,
            "expected_files": list(example.expected_files),
            "expected_symbols": list(example.expected_symbols),
            "candidate_count": len(results),
            "hit_rate": hit_rate_at_k(
                identifiers, relevant, TOP_K
            ),
            "precision": precision_at_k(
                identifiers, relevant, TOP_K
            ),
            "recall": recall_at_k(
                identifiers, relevant, TOP_K
            ),
            "mrr": reciprocal_rank(identifiers, relevant),
            "ndcg": ndcg_at_k(
                identifiers, relevant, TOP_K
            ),
            "top_results": [
                {
                    "rank": rank,
                    "file_path": result.file_path,
                    "symbol": result.symbol,
                    "score": float(result.score),
                    "is_relevant": (
                        benchmark._result_identifier(
                            result,
                            example.query_type,
                        ) in relevant
                    ),
                }
                for rank, result in enumerate(results, start=1)
            ],
        })

    metric_names = (
        "hit_rate",
        "precision",
        "recall",
        "mrr",
        "ndcg",
    )

    overall = {
        metric: (
            sum(item[metric] for item in per_query) / len(per_query)
            if per_query
            else 0.0
        )
        for metric in metric_names
    }

    by_query_type = {}

    for query_type in sorted(
        {item["query_type"] for item in per_query}
    ):
        group = [
            item
            for item in per_query
            if item["query_type"] == query_type
        ]

        by_query_type[query_type] = {
            "query_count": len(group),
            **{
                metric: sum(item[metric] for item in group) / len(group)
                for metric in metric_names
            },
        }

    return {
        "query_count": len(per_query),
        "overall": overall,
        "by_query_type": by_query_type,
        "per_query": per_query,
    }


def evaluate_hybrid_baseline(
    benchmark: RetrievalBenchmark,
    dataset,
    candidate_pools: dict[str, list[RetrievalResult]],
) -> dict:
    """Evaluate hybrid retrieval in its original order, without reranking."""
    rankings = {
        example.query: candidate_pools[example.query]
        for example in dataset
    }
    return evaluate_variant(benchmark, dataset, rankings)


def candidate_pool_diagnostics(
    benchmark: RetrievalBenchmark,
    dataset,
    candidate_pools: dict[str, list[RetrievalResult]],
    top_k: int = TOP_K,
) -> dict:
    """Measure candidate-pool recall and oracle NDCG ceiling per query."""
    per_query = []

    for example in dataset:
        candidates = candidate_pools[example.query]
        identifiers = benchmark._normalise_results(
            candidates, example.query_type
        )
        relevant = benchmark._relevant_ids(example)
        identifiers_set = set(identifiers)
        relevant_in_pool = len(identifiers_set & relevant)
        total_relevant = len(relevant)

        oracle_ranking = (
            list(relevant & identifiers_set)
            + [item for item in identifiers if item not in relevant]
        )

        per_query.append({
            "query": example.query,
            "query_type": example.query_type,
            "candidate_count": len(identifiers),
            "relevant_count": total_relevant,
            "relevant_in_pool": relevant_in_pool,
            "candidate_pool_recall": (
                relevant_in_pool / total_relevant if total_relevant else 0.0
            ),
            "oracle_ndcg": ndcg_at_k(oracle_ranking, relevant, top_k),
        })

    count = len(per_query)
    return {
        "query_count": count,
        "mean_candidate_pool_recall": (
            sum(x["candidate_pool_recall"] for x in per_query) / count
            if count else 0.0
        ),
        "mean_oracle_ndcg": (
            sum(x["oracle_ndcg"] for x in per_query) / count
            if count else 0.0
        ),
        "per_query": per_query,
    }


def main() -> None:
    service = CodeGraphRAGService()

    try:
        # Reuse the existing indexed repository.
        service._is_indexed = True
        bm25_chunk_count = load_bm25_from_qdrant(service)
        print(
            f"Loaded {bm25_chunk_count} chunks into BM25 from Qdrant."
        )

        dataset = list(create_default_dataset())

        if not dataset:
            raise ValueError("The evaluation dataset is empty.")

        config = RetrievalConfig(
            dense_top_k=80,
            bm25_top_k=80,
            graph_top_k=20,
            rrf_top_k=100,
            rerank_top_k=TOP_K,
        )

        adapter = ServiceRetriever(service, config=config)
        pipeline = adapter._hybrid_pipeline(
            include_graph=True,
            include_reranker=True,
        )

        # Warm up the model once.
        pipeline.search(dataset[0].query)

        model = pipeline.reranker.model
        batch_size = pipeline.reranker.batch_size

        candidate_pools: dict[
            str, list[RetrievalResult]
        ] = {}
        candidate_diagnostics: dict[str, dict] = {}

        # Fetch every hybrid candidate pool only once.
        for index, example in enumerate(dataset, start=1):
            retrieved_candidates = pipeline.hybrid.search(
                query=example.query,
                dense_top_k=config.dense_top_k,
                bm25_top_k=config.bm25_top_k,
                graph_top_k=config.graph_top_k,
                rrf_top_k=config.rrf_top_k,
            )

            original_count = len(retrieved_candidates)

            raw_diagnostics = summarize_candidates(
                retrieved_candidates,
                example.expected_files,
                example.expected_symbols,
            )

            candidates = retrieved_candidates

            if example.query_type == "implementation_lookup":
                candidates = unique_file_candidates(candidates)

            candidate_pools[example.query] = candidates

            candidate_diagnostics[example.query] = {
                "query_type": example.query_type,
                "expected_files": list(example.expected_files),
                "expected_symbols": list(example.expected_symbols),
                "retrieved_count": original_count,
                "retained_count": len(candidates),
                "before_deduplication": raw_diagnostics,
                "after_deduplication": summarize_candidates(
                    candidates,
                    example.expected_files,
                    example.expected_symbols,
                ),
            }

            print(
                f"[Candidates {index}/{len(dataset)}] "
                f"{example.query} "
                f"({original_count} retrieved, "
                f"{len(candidates)} retained)"
            )

        benchmark = RetrievalBenchmark(
            dataset=dataset,
            top_k=TOP_K,
        )

        rankings_by_variant: dict[
            str, dict[str, list[RetrievalResult]]
        ] = {
            variant: {}
            for variant in VARIANTS
        }

        # Run the three baseline cross-encoder variants.
        for index, example in enumerate(dataset, start=1):
            candidates = candidate_pools[example.query]

            for variant in BASE_VARIANTS:
                rankings_by_variant[variant][example.query] = (
                    rerank_candidates(
                        model=model,
                        query=example.query,
                        candidates=candidates,
                        variant=variant,
                        batch_size=batch_size,
                    )
                )

            # All weighted variants reuse the same metadata-aware
            # cross-encoder ranking for this query.
            metadata_ranking = rankings_by_variant[
                "path_symbol_code"
            ][example.query]

            for alpha in RRF_WEIGHTS:
                variant_name = (
                    f"weighted_rrf_{int(alpha * 100)}"
                )

                rankings_by_variant[variant_name][example.query] = (
                    reciprocal_rank_fusion(
                        hybrid_candidates=candidates,
                        cross_encoder_ranking=metadata_ranking,
                        alpha=alpha,
                        rrf_k=RRF_K,
                    )
                )

            print(
                f"[Reranking {index}/{len(dataset)}] "
                f"{example.query}"
            )

        results = {
            variant: evaluate_variant(
                benchmark,
                dataset,
                rankings_by_variant[variant],
            )
            for variant in VARIANTS
        }

        results["hybrid_only"] = evaluate_hybrid_baseline(
            benchmark,
            dataset,
            candidate_pools,
        )

        baseline_ndcg = results["code_only"]["overall"]["ndcg"]
        metadata_ndcg = results["path_symbol_code"]["overall"]["ndcg"]

        for variant in VARIANTS:
            ndcg = results[variant]["overall"]["ndcg"]

            results[variant]["ndcg_delta_vs_code_only"] = (
                ndcg - baseline_ndcg
            )
            results[variant]["ndcg_delta_vs_path_symbol_code"] = (
                ndcg - metadata_ndcg
            )

        report = {
            "query_count": len(dataset),
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "methodology": {
                "candidate_pools": (
                    "Each query's hybrid candidate pool is fetched once "
                    "and shared across all variants."
                ),
                "implementation_lookup": (
                    "Candidates are deduplicated by normalized file path "
                    "before reranking."
                ),
                "model": (
                    "The same warmed cross-encoder model is used for "
                    "all three text-format variants. Weighted fusion "
                    "reuses the path_symbol_code ranking."
                ),
                "weighted_rrf": {
                    "formula": (
                        "alpha / (RRF_K + cross_encoder_rank) + "
                        "(1 - alpha) / (RRF_K + hybrid_rank)"
                    ),
                    "cross_encoder_variant": "path_symbol_code",
                    "rrf_k": RRF_K,
                    "tested_cross_encoder_weights": list(RRF_WEIGHTS),
                    "score_note": (
                        "Fusion uses ranks rather than raw scores. "
                        "Alpha 1.0 should reproduce the cross-encoder "
                        "ordering, subject to ties and tie-breaking."
                    ),
                },
                "variants": {
                    "code_only": "Code content only",
                    "path_code": "File path followed by code content",
                    "path_symbol_code": (
                        "File path, symbol, then code content"
                    ),
                    **{
                        f"weighted_rrf_{int(alpha * 100)}": (
                            f"{int(alpha * 100)}% cross-encoder rank, "
                            f"{int((1 - alpha) * 100)}% hybrid rank"
                        )
                        for alpha in RRF_WEIGHTS
                    },
                },
                "ranking_output": (
                    "The complete candidate rankings are saved. "
                    "Hit rate, precision, and NDCG use TOP_K; recall "
                    "and MRR follow their metric definitions."
                ),
                "candidate_diagnostics": (
                    "Original hybrid rankings and scores are recorded "
                    "before and after deduplication."
                ),
                "hybrid_baseline": (
                    "hybrid_only evaluates the original hybrid candidate "
                    "order before cross-encoder reranking."
                ),
                "candidate_pool_diagnostics": (
                    "Candidate-pool recall measures how many labeled "
                    "relevant identifiers are present before reranking. "
                    "Oracle NDCG@5 is the best binary-relevance NDCG@5 "
                    "possible using only those candidates and labels."
                ),
                "limitations": (
                    "Results use the existing 31-query development dataset "
                    "and its current relevance labels. Further held-out "
                    "evaluation is needed before choosing a production "
                    "configuration."
                ),
            },
            "candidate_diagnostics": candidate_diagnostics,
            "candidate_pool_diagnostics": candidate_pool_diagnostics(
                benchmark, dataset, candidate_pools, TOP_K
            ),
            "results": results,
        }

        output = Path(
            "evaluation/results/reranker_ablation.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\n=== Ablation results ===")

        print(
            f"\n{'Variant':<24}"
            f"{'NDCG@5':>10}"
            f"{'MRR':>10}"
            f"{'Hit@5':>10}"
            f"{'Recall@5':>12}"
        )

        for variant in ("hybrid_only", *VARIANTS):
            data = results[variant]
            metrics = data["overall"]

            print(
                f"{variant:<24}"
                f"{metrics['ndcg']:>10.4f}"
                f"{metrics['mrr']:>10.4f}"
                f"{metrics['hit_rate']:>10.4f}"
                f"{metrics['recall']:>12.4f}"
            )

            if variant != "hybrid_only":
                print(
                    f"  NDCG delta vs code_only: "
                    f"{data['ndcg_delta_vs_code_only']:+.4f}"
                )
                print(
                    f"  NDCG delta vs path_symbol_code: "
                    f"{data['ndcg_delta_vs_path_symbol_code']:+.4f}"
                )

        pool_metrics = report["candidate_pool_diagnostics"]
        print(
            "\nCandidate-pool mean Recall: "
            f"{pool_metrics['mean_candidate_pool_recall']:.4f}"
        )
        print(
            "Candidate-pool oracle NDCG@5: "
            f"{pool_metrics['mean_oracle_ndcg']:.4f}"
        )

        print(f"\nSaved full report to: {output}")
        print("Full rankings and candidate diagnostics are included.")

    finally:
        service.close()


if __name__ == "__main__":
    main()

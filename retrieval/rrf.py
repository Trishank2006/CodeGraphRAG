from dataclasses import replace

from retrieval.models import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """
    Combine ranked retrieval results using Reciprocal Rank Fusion (RRF).

    RRF score:
        score(d) = sum(1 / (k + rank(d)))

    Rank is 1-based within each input result list.

    Results with the same `id` are deduplicated. Their sources are
    combined into a deterministic `+`-separated source string.

    Args:
        result_lists: Ranked result lists from different retrievers.
        k: RRF constant. Larger values reduce the effect of rank position.
        top_k: Maximum number of fused results to return.

    Returns:
        Fused and deterministically ordered RetrievalResult objects.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    merged_results: dict[str, RetrievalResult] = {}
    sources: dict[str, list[str]] = {}

    for results in result_lists:
        seen_in_list: set[str] = set()

        for rank, result in enumerate(results, start=1):
            # Do not count the same document twice within one retriever.
            if result.id in seen_in_list:
                continue

            seen_in_list.add(result.id)

            scores[result.id] = scores.get(result.id, 0.0) + (
                1.0 / (k + rank)
            )

            if result.id not in merged_results:
                merged_results[result.id] = result

            source_list = sources.setdefault(result.id, [])
            if result.source not in source_list:
                source_list.append(result.source)

    fused_results: list[RetrievalResult] = []

    for result_id, result in merged_results.items():
        combined_source = "+".join(sources[result_id])

        fused_results.append(
            replace(
                result,
                source=combined_source,
                score=scores[result_id],
            )
        )

    # Deterministic ordering:
    # 1. Highest RRF score first
    # 2. ID alphabetically for ties
    fused_results.sort(
        key=lambda result: (-result.score, result.id)
    )

    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        fused_results = fused_results[:top_k]

    return fused_results

from dataclasses import replace

from retrieval.models import RetrievalResult


def _merge_duplicate_results(
    existing: RetrievalResult,
    incoming: RetrievalResult,
) -> RetrievalResult:
    """Merge metadata for the same document without losing useful content."""

    def choose_text(first: str, second: str) -> str:
        first = first or ""
        second = second or ""
        return second if len(second.strip()) > len(first.strip()) else first

    def choose_int(first: int, second: int) -> int:
        return first if first > 0 else second

    return replace(
        existing,
        source=existing.source,
        repository=(
            existing.repository
            if existing.repository != "default_repo"
            else incoming.repository
        ),
        file_path=existing.file_path or incoming.file_path,
        language=(
            existing.language
            if existing.language and existing.language != "unknown"
            else incoming.language
        ),
        symbol=existing.symbol or incoming.symbol,
        start_line=choose_int(existing.start_line, incoming.start_line),
        end_line=choose_int(existing.end_line, incoming.end_line),
        content=choose_text(existing.content, incoming.content),
    )


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """
    Combine ranked retrieval lists using Reciprocal Rank Fusion.

    Documents are deduplicated by exact ID. Duplicate results contribute
    once per input list, while useful metadata is merged across sources.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")

    scores: dict[str, float] = {}
    merged_results: dict[str, RetrievalResult] = {}
    sources: dict[str, list[str]] = {}

    for results in result_lists:
        seen_in_list: set[str] = set()

        for rank, result in enumerate(results, start=1):
            result_id = result.id

            if not result_id or result_id in seen_in_list:
                continue

            seen_in_list.add(result_id)
            scores[result_id] = scores.get(result_id, 0.0) + (
                1.0 / (k + rank)
            )

            if result_id not in merged_results:
                merged_results[result_id] = result
            else:
                merged_results[result_id] = _merge_duplicate_results(
                    merged_results[result_id],
                    result,
                )

            source_list = sources.setdefault(result_id, [])
            for source in (result.source or "").split("+"):
                if source and source not in source_list:
                    source_list.append(source)

    fused_results: list[RetrievalResult] = []

    for result_id, result in merged_results.items():
        fused_results.append(
            replace(
                result,
                source="+".join(sources[result_id]),
                score=scores[result_id],
            )
        )

    fused_results.sort(
        key=lambda result: (-result.score, result.id)
    )

    if top_k is not None:
        fused_results = fused_results[:top_k]

    return fused_results

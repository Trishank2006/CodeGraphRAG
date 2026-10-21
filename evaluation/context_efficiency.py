
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ContextEfficiencyStats:
    """Statistics describing retrieved context size."""

    sample_count: int
    average_characters: float
    average_chunks: float
    average_tokens: float
    max_characters: int
    max_tokens: int


def estimate_tokens(text: str) -> int:
    """Estimate tokens deterministically at approximately four characters each."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def context_statistics(
    contexts: Sequence[Sequence[str]] | Iterable[Sequence[str]],
) -> ContextEfficiencyStats:
    """Calculate context-size statistics across multiple samples."""
    context_list = list(contexts)

    if not context_list:
        return ContextEfficiencyStats(0, 0.0, 0.0, 0.0, 0, 0)

    character_counts: list[int] = []
    chunk_counts: list[int] = []
    token_counts: list[int] = []

    for context in context_list:
        chunks = [str(chunk) for chunk in context]
        text = "\n".join(chunks)

        character_counts.append(len(text))
        chunk_counts.append(len(chunks))
        token_counts.append(estimate_tokens(text))

    count = len(context_list)
    return ContextEfficiencyStats(
        sample_count=count,
        average_characters=sum(character_counts) / count,
        average_chunks=sum(chunk_counts) / count,
        average_tokens=sum(token_counts) / count,
        max_characters=max(character_counts),
        max_tokens=max(token_counts),
    )


def apply_context_budget(
    chunks: Sequence[str],
    max_chunks: int | None = None,
    max_characters: int | None = None,
) -> list[str]:
    """Apply chunk and character budgets while preserving order."""
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0")
    if max_characters is not None and max_characters <= 0:
        raise ValueError("max_characters must be greater than 0")

    selected: list[str] = []
    current_characters = 0

    for chunk in chunks:
        chunk = str(chunk)

        if max_chunks is not None and len(selected) >= max_chunks:
            break

        additional = len(chunk) + (1 if selected else 0)
        if (
            max_characters is not None
            and current_characters + additional > max_characters
        ):
            break

        selected.append(chunk)
        current_characters += additional

    return selected






def compare_context_budgets(
    chunks: Sequence[str] | Sequence[Sequence[str]],
    budgets: Sequence[int],
) -> dict[int, ContextEfficiencyStats]:
    """Compare character budgets across chunks or per-query chunk lists."""
    results: dict[int, ContextEfficiencyStats] = {}

    if not chunks:
        samples: list[list[str]] = []
    elif isinstance(chunks[0], str):
        # Backward-compatible input: a flat list of chunks.
        samples = [list(chunks)]
    else:
        # Each inner list represents one query's retrieved chunks.
        samples = [list(sample) for sample in chunks]

    for budget in budgets:
        if budget <= 0:
            raise ValueError("budgets must contain positive values")

        budgeted_samples = [
            apply_context_budget(
                sample,
                max_characters=budget,
            )
            for sample in samples
        ]
        results[budget] = context_statistics(budgeted_samples)

    return results






def stats_to_dict(
    stats: ContextEfficiencyStats,
) -> dict[str, float | int]:
    """Convert statistics into a JSON-friendly dictionary."""
    return {
        "sample_count": stats.sample_count,
        "average_characters": stats.average_characters,
        "average_chunks": stats.average_chunks,
        "average_tokens": stats.average_tokens,
        "max_characters": stats.max_characters,
        "max_tokens": stats.max_tokens,
    }

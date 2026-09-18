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
    """
    Estimate token count using a simple character-based approximation.

    This is intentionally deterministic and dependency-free.
    A rough estimate of four characters per token is used.
    """
    if not text:
        return 0

    return max(1, (len(text) + 3) // 4)


def context_statistics(
    contexts: Sequence[Sequence[str]] | Iterable[Sequence[str]],
) -> ContextEfficiencyStats:
    """Calculate context-size statistics across multiple samples."""

    context_list = list(contexts)

    if not context_list:
        return ContextEfficiencyStats(
            sample_count=0,
            average_characters=0.0,
            average_chunks=0.0,
            average_tokens=0.0,
            max_characters=0,
            max_tokens=0,
        )

    character_counts: list[int] = []
    chunk_counts: list[int] = []
    token_counts: list[int] = []

    for context in context_list:
        chunks = [str(chunk) for chunk in context]

        text = "\n".join(chunks)

        character_counts.append(len(text))
        chunk_counts.append(len(chunks))
        token_counts.append(estimate_tokens(text))

    return ContextEfficiencyStats(
        sample_count=len(context_list),
        average_characters=sum(character_counts) / len(character_counts),
        average_chunks=sum(chunk_counts) / len(chunk_counts),
        average_tokens=sum(token_counts) / len(token_counts),
        max_characters=max(character_counts),
        max_tokens=max(token_counts),
    )


def apply_context_budget(
    chunks: Sequence[str],
    max_chunks: int | None = None,
    max_characters: int | None = None,
) -> list[str]:
    """
    Apply chunk and character budgets while preserving order.

    A chunk that would exceed the character budget is not included.
    """
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

        additional_characters = len(chunk)

        if selected:
            additional_characters += 1

        if (
            max_characters is not None
            and current_characters + additional_characters > max_characters
        ):
            break

        selected.append(chunk)
        current_characters += additional_characters

    return selected


def compare_context_budgets(
    chunks: Sequence[str],
    budgets: Sequence[int],
) -> dict[int, ContextEfficiencyStats]:
    """
    Compare context statistics for multiple character budgets.
    """
    results: dict[int, ContextEfficiencyStats] = {}

    for budget in budgets:
        if budget <= 0:
            raise ValueError("budgets must contain positive values")

        context = apply_context_budget(
            chunks,
            max_characters=budget,
        )

        results[budget] = context_statistics([context])

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
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _normalise(items: Iterable[str]) -> set[str]:
    return {str(item) for item in items}


def hit_rate_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Return 1.0 if at least one relevant item appears in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_set = _normalise(relevant)

    if not relevant_set:
        return 0.0

    return float(bool(set(retrieved[:k]) & relevant_set))


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_set = _normalise(relevant)
    top_k = list(retrieved[:k])

    if not top_k:
        return 0.0

    relevant_count = sum(item in relevant_set for item in top_k)
    return relevant_count / len(top_k)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Fraction of all relevant items found in the top-k results."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_set = _normalise(relevant)

    if not relevant_set:
        return 0.0

    found = set(retrieved[:k]) & relevant_set
    return len(found) / len(relevant_set)


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Iterable[str],
) -> float:
    """Return reciprocal rank of the first relevant result."""
    relevant_set = _normalise(relevant)

    if not relevant_set:
        return 0.0

    for index, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / index

    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[Iterable[str]],
) -> float:
    """Return mean reciprocal rank across multiple queries."""
    if len(rankings) != len(relevant_sets):
        raise ValueError("rankings and relevant_sets must have the same length")

    if not rankings:
        return 0.0

    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(rankings, relevant_sets)
    ]

    return sum(scores) / len(scores)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """
    Calculate binary-relevance nDCG@k.

    Relevant items receive gain 1.
    Non-relevant items receive gain 0.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_set = _normalise(relevant)

    if not relevant_set:
        return 0.0

    top_k = retrieved[:k]

    dcg = 0.0

    for rank, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_count = min(len(relevant_set), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
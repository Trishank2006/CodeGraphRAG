from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence


def _tokens(text: str) -> list[str]:
    """Convert text into normalized word/code tokens."""
    return re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _overlap_score(
    source: str,
    target: str,
) -> float:
    """
    Calculate token overlap from target against source.

    Returns a value between 0 and 1.
    """
    target_tokens = _token_set(target)

    if not target_tokens:
        return 0.0

    source_tokens = _token_set(source)

    return len(target_tokens & source_tokens) / len(target_tokens)


def faithfulness(
    answer: str,
    context: Sequence[str] | Iterable[str],
) -> float:
    """
    Estimate how much of the answer is supported by the supplied context.

    This is a lightweight lexical metric intended for deterministic
    offline benchmarking. It does not claim semantic equivalence.
    """
    if not answer.strip():
        return 0.0

    context_text = "\n".join(str(item) for item in context)

    if not context_text.strip():
        return 0.0

    return _overlap_score(context_text, answer)


def answer_relevance(
    query: str,
    answer: str,
) -> float:
    """
    Estimate how relevant an answer is to a query using token overlap.

    Returns a value between 0 and 1.
    """
    if not query.strip() or not answer.strip():
        return 0.0

    return _overlap_score(answer, query)


def context_relevance(
    query: str,
    context: Sequence[str] | Iterable[str],
) -> float:
    """
    Estimate how relevant retrieved context is to the user query.

    The score is the average query-token coverage across the supplied
    context as a whole.
    """
    if not query.strip():
        return 0.0

    context_text = "\n".join(str(item) for item in context)

    if not context_text.strip():
        return 0.0

    return _overlap_score(context_text, query)


def generation_metrics(
    query: str,
    answer: str,
    context: Sequence[str] | Iterable[str],
) -> dict[str, float]:
    """Calculate all generation metrics for one query."""

    return {
        "faithfulness": faithfulness(answer, context),
        "answer_relevance": answer_relevance(query, answer),
        "context_relevance": context_relevance(query, context),
    }


def average_generation_metrics(
    metrics: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Average generation metrics across multiple evaluation examples."""

    if not metrics:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_relevance": 0.0,
        }

    keys = (
        "faithfulness",
        "answer_relevance",
        "context_relevance",
    )

    return {
        key: sum(item.get(key, 0.0) for item in metrics) / len(metrics)
        for key in keys
    }
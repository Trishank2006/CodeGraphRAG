import pytest

from evaluation.context_efficiency import (
    apply_context_budget,
    compare_context_budgets,
    context_statistics,
    estimate_tokens,
    stats_to_dict,
)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("abcde") == 2


def test_context_statistics():
    contexts = [
        ["abcd", "efgh"],
        ["12345678"],
    ]

    stats = context_statistics(contexts)

    assert stats.sample_count == 2
    assert stats.average_characters == 8.5
    assert stats.average_chunks == 1.5
    assert stats.average_tokens == 2.5
    assert stats.max_characters == 9
    assert stats.max_tokens == 3


def test_empty_context_statistics():
    stats = context_statistics([])

    assert stats.sample_count == 0
    assert stats.average_characters == 0.0
    assert stats.average_chunks == 0.0
    assert stats.average_tokens == 0.0
    assert stats.max_characters == 0
    assert stats.max_tokens == 0


def test_apply_context_budget_by_chunks():
    chunks = [
        "chunk one",
        "chunk two",
        "chunk three",
    ]

    result = apply_context_budget(
        chunks,
        max_chunks=2,
    )

    assert result == [
        "chunk one",
        "chunk two",
    ]


def test_apply_context_budget_by_characters():
    chunks = [
        "1234",
        "5678",
        "90",
    ]

    result = apply_context_budget(
        chunks,
        max_characters=9,
    )

    assert result == [
        "1234",
        "5678",
    ]


def test_apply_context_budget_preserves_order():
    chunks = [
        "first",
        "second",
        "third",
    ]

    result = apply_context_budget(
        chunks,
        max_characters=20,
    )

    assert result == chunks


def test_context_budget_empty_chunks():
    assert apply_context_budget([]) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chunks": 0},
        {"max_characters": 0},
    ],
)
def test_invalid_context_budget(kwargs):
    with pytest.raises(ValueError):
        apply_context_budget(["test"], **kwargs)


def test_compare_context_budgets():
    chunks = [
        "a" * 10,
        "b" * 10,
        "c" * 10,
    ]

    results = compare_context_budgets(
        chunks,
        budgets=[10, 25, 50],
    )

    assert list(results) == [10, 25, 50]

    assert results[10].average_chunks == 1
    assert results[25].average_chunks == 2
    assert results[50].average_chunks == 3


def test_invalid_context_comparison_budget():
    with pytest.raises(ValueError):
        compare_context_budgets(
            ["test"],
            budgets=[10, 0, 20],
        )


def test_stats_to_dict():
    stats = context_statistics(
        [["abcdefgh"]],
    )

    data = stats_to_dict(stats)

    assert data["sample_count"] == 1
    assert data["average_characters"] == 8.0
    assert data["average_chunks"] == 1.0
    assert data["average_tokens"] == 2.0
    assert data["max_characters"] == 8
    assert data["max_tokens"] == 2
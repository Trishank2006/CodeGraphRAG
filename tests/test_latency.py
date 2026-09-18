import pytest

from evaluation.latency import (
    benchmark_callable,
    benchmark_pipeline_stages,
    calculate_latency_stats,
    latency_stats_to_dict,
    measure_callable,
    percentile,
)


def test_percentile_single_value():
    assert percentile([10.0], 50) == 10.0
    assert percentile([10.0], 95) == 10.0


def test_percentile_interpolation():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == pytest.approx(48.0)
    assert percentile(values, 99) == pytest.approx(49.6)


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_percentile_invalid():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -1)

    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)


def test_calculate_latency_stats():
    stats = calculate_latency_stats(
        [10.0, 20.0, 30.0, 40.0, 50.0]
    )

    assert stats.count == 5
    assert stats.mean_ms == 30.0
    assert stats.p50_ms == 30.0
    assert stats.p95_ms == pytest.approx(48.0)
    assert stats.p99_ms == pytest.approx(49.6)
    assert stats.min_ms == 10.0
    assert stats.max_ms == 50.0


def test_empty_latency_stats():
    stats = calculate_latency_stats([])

    assert stats.count == 0
    assert stats.mean_ms == 0.0
    assert stats.p50_ms == 0.0
    assert stats.p95_ms == 0.0
    assert stats.p99_ms == 0.0


def test_negative_latency_rejected():
    with pytest.raises(ValueError):
        calculate_latency_stats([10.0, -1.0])


def test_measure_callable():
    durations, outputs = measure_callable(
        lambda: 42,
        iterations=3,
    )

    assert len(durations) == 3
    assert outputs == [42, 42, 42]
    assert all(duration >= 0.0 for duration in durations)


def test_invalid_iterations():
    with pytest.raises(ValueError):
        measure_callable(lambda: None, iterations=0)


def test_benchmark_callable():
    name, stats = benchmark_callable(
        "Dense",
        lambda: [1, 2, 3],
        iterations=3,
    )

    assert name == "Dense"
    assert stats.count == 3
    assert stats.mean_ms >= 0.0
    assert stats.p50_ms >= 0.0
    assert stats.p95_ms >= 0.0
    assert stats.p99_ms >= 0.0


def test_invalid_benchmark_name():
    with pytest.raises(ValueError):
        benchmark_callable(
            "",
            lambda: None,
        )


def test_latency_stats_to_dict():
    stats = calculate_latency_stats([10.0, 20.0, 30.0])

    data = latency_stats_to_dict(stats)

    assert data["count"] == 3
    assert data["mean_ms"] == 20.0
    assert data["p50_ms"] == 20.0
    assert data["min_ms"] == 10.0
    assert data["max_ms"] == 30.0


def test_benchmark_pipeline_stages():
    results = benchmark_pipeline_stages(
        {
            "Dense": lambda: "dense",
            "BM25": lambda: "bm25",
            "RRF": lambda: "rrf",
            "Reranker": lambda: "reranked",
        },
        iterations=2,
    )

    assert set(results) == {
        "Dense",
        "BM25",
        "RRF",
        "Reranker",
    }

    for stats in results.values():
        assert stats["count"] == 2
        assert stats["mean_ms"] >= 0.0
        assert stats["p50_ms"] >= 0.0
        assert stats["p95_ms"] >= 0.0
        assert stats["p99_ms"] >= 0.0
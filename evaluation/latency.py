from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class LatencyStats:
    """Latency statistics in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * (percentile_value / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * weight
    )


def calculate_latency_stats(
    durations_ms: Sequence[float],
) -> LatencyStats:
    """Calculate standard latency statistics."""
    if not durations_ms:
        return LatencyStats(
            count=0,
            mean_ms=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
        )

    if any(duration < 0 for duration in durations_ms):
        raise ValueError("latencies cannot be negative")

    return LatencyStats(
        count=len(durations_ms),
        mean_ms=sum(durations_ms) / len(durations_ms),
        p50_ms=percentile(durations_ms, 50),
        p95_ms=percentile(durations_ms, 95),
        p99_ms=percentile(durations_ms, 99),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
    )


def measure_callable(
    function: Callable[[], object],
    iterations: int = 1,
) -> tuple[list[float], list[object]]:
    """Measure a callable repeatedly and return durations in milliseconds."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than 0")

    durations: list[float] = []
    outputs: list[object] = []

    for _ in range(iterations):
        start = time.perf_counter()

        output = function()

        elapsed = time.perf_counter() - start

        durations.append(elapsed * 1000.0)
        outputs.append(output)

    return durations, outputs


def benchmark_callable(
    name: str,
    function: Callable[[], object],
    iterations: int = 10,
) -> tuple[str, LatencyStats]:
    """Benchmark a callable and return named latency statistics."""
    if not name.strip():
        raise ValueError("name must not be empty")

    durations, _ = measure_callable(
        function,
        iterations=iterations,
    )

    return name, calculate_latency_stats(durations)


def latency_stats_to_dict(
    stats: LatencyStats,
) -> dict[str, float | int]:
    """Convert latency statistics to a JSON-friendly dictionary."""
    return {
        "count": stats.count,
        "mean_ms": stats.mean_ms,
        "p50_ms": stats.p50_ms,
        "p95_ms": stats.p95_ms,
        "p99_ms": stats.p99_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
    }


def benchmark_pipeline_stages(
    stages: dict[str, Callable[[], object]],
    iterations: int = 10,
) -> dict[str, dict[str, float | int]]:
    """Benchmark multiple named pipeline stages."""
    results: dict[str, dict[str, float | int]] = {}

    for name, function in stages.items():
        _, stats = benchmark_callable(
            name,
            function,
            iterations=iterations,
        )

        results[name] = latency_stats_to_dict(stats)

    return results
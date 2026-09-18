from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AblationResult:
    """Metrics for one ablation configuration."""

    name: str
    recall: float
    mrr: float
    ndcg: float
    faithfulness: float
    citation_accuracy: float
    latency_ms: float


@dataclass(frozen=True)
class AblationComparison:
    """Comparison between an ablation configuration and a baseline."""

    name: str
    recall_delta: float
    mrr_delta: float
    ndcg_delta: float
    faithfulness_delta: float
    citation_accuracy_delta: float
    latency_delta_ms: float


ABLATION_STAGES: tuple[str, ...] = (
    "Dense",
    "Dense + BM25",
    "Dense + BM25 + RRF",
    "Dense + BM25 + RRF + Reranker",
    "Full System",
)


def validate_ablation_order(
    names: Sequence[str],
) -> None:
    """Validate that configurations follow the expected ablation order."""

    expected = list(ABLATION_STAGES)

    if list(names) != expected:
        raise ValueError(
            "Ablation configurations must follow the expected order: "
            + " -> ".join(expected)
        )


def create_ablation_result(
    name: str,
    *,
    recall: float,
    mrr: float,
    ndcg: float,
    faithfulness: float,
    citation_accuracy: float,
    latency_ms: float,
) -> AblationResult:
    """Create and validate one ablation result."""

    if not name.strip():
        raise ValueError("name must not be empty")

    metrics = {
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "faithfulness": faithfulness,
        "citation_accuracy": citation_accuracy,
    }

    for metric_name, value in metrics.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{metric_name} must be between 0 and 1"
            )

    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")

    return AblationResult(
        name=name,
        recall=recall,
        mrr=mrr,
        ndcg=ndcg,
        faithfulness=faithfulness,
        citation_accuracy=citation_accuracy,
        latency_ms=latency_ms,
    )


def compare_to_baseline(
    results: Sequence[AblationResult],
    baseline_name: str,
) -> list[AblationComparison]:
    """
    Compare every configuration against the selected baseline.
    """

    if not results:
        return []

    baseline = next(
        (
            result
            for result in results
            if result.name == baseline_name
        ),
        None,
    )

    if baseline is None:
        raise ValueError(
            f"Baseline '{baseline_name}' was not found"
        )

    comparisons: list[AblationComparison] = []

    for result in results:
        comparisons.append(
            AblationComparison(
                name=result.name,
                recall_delta=result.recall - baseline.recall,
                mrr_delta=result.mrr - baseline.mrr,
                ndcg_delta=result.ndcg - baseline.ndcg,
                faithfulness_delta=(
                    result.faithfulness - baseline.faithfulness
                ),
                citation_accuracy_delta=(
                    result.citation_accuracy
                    - baseline.citation_accuracy
                ),
                latency_delta_ms=(
                    result.latency_ms - baseline.latency_ms
                ),
            )
        )

    return comparisons


def ablation_to_dict(
    result: AblationResult,
) -> dict[str, float | str]:
    """Convert an ablation result to a JSON-friendly dictionary."""

    return {
        "name": result.name,
        "recall": result.recall,
        "mrr": result.mrr,
        "ndcg": result.ndcg,
        "faithfulness": result.faithfulness,
        "citation_accuracy": result.citation_accuracy,
        "latency_ms": result.latency_ms,
    }


def comparison_to_dict(
    comparison: AblationComparison,
) -> dict[str, float | str]:
    """Convert an ablation comparison to a JSON-friendly dictionary."""

    return {
        "name": comparison.name,
        "recall_delta": comparison.recall_delta,
        "mrr_delta": comparison.mrr_delta,
        "ndcg_delta": comparison.ndcg_delta,
        "faithfulness_delta": comparison.faithfulness_delta,
        "citation_accuracy_delta": comparison.citation_accuracy_delta,
        "latency_delta_ms": comparison.latency_delta_ms,
    }


def results_to_table(
    results: Iterable[AblationResult],
) -> list[dict[str, float | str]]:
    """Convert ablation results into rows suitable for reporting."""

    return [
        ablation_to_dict(result)
        for result in results
    ]
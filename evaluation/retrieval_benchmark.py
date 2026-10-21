
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Sequence

from evaluation.dataset import EvaluationExample
from evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics produced by one retrieval configuration."""

    name: str
    hit_rate: float
    precision: float
    recall: float
    mrr: float
    ndcg: float


class RetrievalBenchmark:
    """Evaluate retrieval configurations against an evaluation dataset."""

    def __init__(
        self,
        dataset: Iterable[EvaluationExample],
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.dataset = tuple(dataset)
        self.top_k = top_k
        self.validate_dataset()

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Normalize repository paths to POSIX format."""
        return str(PurePosixPath(path.replace("\\", "/")))

    @staticmethod
    def _relevant_ids(example: EvaluationExample) -> set[str]:
        """Return ground-truth identifiers matching the result identifier type."""
        if example.query_type == "symbol_lookup":
            return {
                symbol.strip()
                for symbol in example.expected_symbols
                if symbol and symbol.strip()
            }

        return {
            RetrievalBenchmark._normalise_path(path)
            for path in example.expected_files
            if path and path.strip()
        }

    def validate_dataset(self) -> None:
        """Reject examples without relevant labels."""
        for index, example in enumerate(self.dataset):
            if not example.query.strip():
                raise ValueError(
                    f"Evaluation example {index} has an empty query"
                )

            relevant = self._relevant_ids(example)

            if not relevant:
                raise ValueError(
                    f"Evaluation example {index} has no relevant labels: "
                    f"{example.query!r} "
                    f"(type={example.query_type!r})"
                )

    @staticmethod
    def _result_identifier(
        result: Any,
        query_type: str,
    ) -> str:
        """Extract a symbol or file identifier according to query type."""
        symbol_lookup = query_type == "symbol_lookup"

        if isinstance(result, str):
            value = result.strip()

            if not value:
                raise ValueError("Retriever returned an empty identifier")

            if "\\" in value or "/" in value:
                return RetrievalBenchmark._normalise_path(value)

            return value

        if isinstance(result, dict):
            symbol = result.get("symbol")
            file_path = result.get("file_path")
            result_id = result.get("id")
        else:
            symbol = getattr(result, "symbol", None)
            file_path = getattr(result, "file_path", None)
            result_id = getattr(result, "id", None)

        if symbol_lookup:
            if symbol and str(symbol).strip():
                return str(symbol).strip()

            if result_id and str(result_id).strip():
                return str(result_id).strip()
        else:
            if file_path and str(file_path).strip():
                return RetrievalBenchmark._normalise_path(
                    str(file_path).strip()
                )

            if result_id and str(result_id).strip():
                return str(result_id).strip()

        raise TypeError(
            "Retriever result must be a string, dictionary, "
            "or object containing a suitable symbol/file_path/id"
        )

    def _normalise_results(
        self,
        results: Sequence[Any] | Iterable[Any],
        query_type: str,
    ) -> list[str]:
        """Normalize and deduplicate results while preserving ranking order."""
        identifiers: list[str] = []
        seen: set[str] = set()

        for result in results:
            identifier = self._result_identifier(result, query_type)

            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)

        return identifiers

    @staticmethod
    def _calculate_metrics(
        rankings: list[list[str]],
        relevant_sets: list[set[str]],
        hit_scores: list[float],
        precision_scores: list[float],
        recall_scores: list[float],
        ndcg_scores: list[float],
    ) -> dict[str, float | int]:
        """Calculate aggregate metrics for a group of queries."""
        if not rankings:
            return {
                "query_count": 0,
                "hit_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
            }

        return {
            "query_count": len(rankings),
            "hit_rate": sum(hit_scores) / len(hit_scores),
            "precision": sum(precision_scores) / len(precision_scores),
            "recall": sum(recall_scores) / len(recall_scores),
            "mrr": mean_reciprocal_rank(rankings, relevant_sets),
            "ndcg": sum(ndcg_scores) / len(ndcg_scores),
        }

    def evaluate_detailed(
        self,
        name: str,
        retriever: Callable[[EvaluationExample], Sequence[Any]],
    ) -> dict[str, Any]:
        """
        Evaluate one configuration and return overall and per-query-type
        metrics without running the retriever more than once per example.
        """
        if not name.strip():
            raise ValueError("name must not be empty")

        groups: dict[str, dict[str, list[Any]]] = {}
        per_query: list[dict[str, Any]] = []

        for example in self.dataset:
            raw_results = retriever(example)

            retrieved = self._normalise_results(
                raw_results,
                example.query_type,
            )
            relevant = self._relevant_ids(example)
            first_relevant_rank = next(
                (
                    rank
                    for rank, identifier in enumerate(retrieved, start=1)
                    if identifier in relevant
                ),
                None,
            )

            per_query.append(
                {
                    "query": example.query,
                    "query_type": example.query_type,
                    "expected_targets": sorted(relevant),
                    "ranked_results": retrieved[: self.top_k],
                    "hit": (
                        first_relevant_rank is not None
                        and first_relevant_rank <= self.top_k
                    ),
                    "first_relevant_rank": first_relevant_rank,
                }
            )

            hit = hit_rate_at_k(retrieved, relevant, self.top_k)
            precision = precision_at_k(
                retrieved,
                relevant,
                self.top_k,
            )
            recall = recall_at_k(
                retrieved,
                relevant,
                self.top_k,
            )
            ndcg = ndcg_at_k(
                retrieved,
                relevant,
                self.top_k,
            )

            for group_name in ("overall", example.query_type):
                group = groups.setdefault(
                    group_name,
                    {
                        "rankings": [],
                        "relevant_sets": [],
                        "hit_scores": [],
                        "precision_scores": [],
                        "recall_scores": [],
                        "ndcg_scores": [],
                    },
                )

                group["rankings"].append(retrieved)
                group["relevant_sets"].append(relevant)
                group["hit_scores"].append(hit)
                group["precision_scores"].append(precision)
                group["recall_scores"].append(recall)
                group["ndcg_scores"].append(ndcg)

        metrics: dict[str, dict[str, float | int]] = {}

        for group_name, group in groups.items():
            metrics[group_name] = self._calculate_metrics(
                group["rankings"],
                group["relevant_sets"],
                group["hit_scores"],
                group["precision_scores"],
                group["recall_scores"],
                group["ndcg_scores"],
            )

        # Preserve the empty-dataset behavior expected by existing tests.
        if "overall" not in metrics:
            metrics["overall"] = self._calculate_metrics(
                [], [], [], [], [], []
            )

        return {
            "name": name,
            "by_query_type": metrics,
            "per_query": per_query,
        }

    def evaluate(
        self,
        name: str,
        retriever: Callable[[EvaluationExample], Sequence[Any]],
    ) -> BenchmarkResult:
        """Evaluate one retrieval configuration using overall metrics."""
        detailed = self.evaluate_detailed(name, retriever)
        overall = detailed["by_query_type"]["overall"]

        return BenchmarkResult(
            name=name,
            hit_rate=overall["hit_rate"],
            precision=overall["precision"],
            recall=overall["recall"],
            mrr=overall["mrr"],
            ndcg=overall["ndcg"],
        )

    def compare(
        self,
        retrievers: dict[
            str,
            Callable[[EvaluationExample], Sequence[Any]],
        ],
    ) -> list[BenchmarkResult]:
        """Evaluate multiple retrieval configurations."""
        return [
            self.evaluate(name, retriever)
            for name, retriever in retrievers.items()
        ]


def results_to_dict(
    results: Sequence[BenchmarkResult],
) -> list[dict[str, float | str]]:
    """Convert benchmark results into JSON-friendly dictionaries."""
    return [
        {
            "name": result.name,
            "hit_rate": result.hit_rate,
            "precision": result.precision,
            "recall": result.recall,
            "mrr": result.mrr,
            "ndcg": result.ndcg,
        }
        for result in results
    ]

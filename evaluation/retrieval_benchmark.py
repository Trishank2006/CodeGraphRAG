from __future__ import annotations

from dataclasses import dataclass
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
    """
    Evaluate retrieval configurations against an evaluation dataset.

    Retrievers may return:
      - strings
      - RetrievalResult objects
      - dictionaries containing file_path/id fields

    File paths are used as the primary relevance identifier.
    """

    def __init__(
        self,
        dataset: Iterable[EvaluationExample],
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.dataset = tuple(dataset)
        self.top_k = top_k

    @staticmethod
    def _relevant_ids(
        example: EvaluationExample,
    ) -> set[str]:
        """Build the ground-truth relevance set."""

        relevant: set[str] = set(example.expected_files)

        for symbol in example.expected_symbols:
            relevant.add(symbol)

        return relevant

    @staticmethod
    def _result_identifier(result: Any) -> str:
        """
        Extract a stable evaluation identifier from a retrieval result.

        File path is preferred because the evaluation dataset stores
        expected repository files.
        """

        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            file_path = result.get("file_path")

            if file_path:
                return str(file_path)

            result_id = result.get("id")

            if result_id:
                return str(result_id)

        file_path = getattr(result, "file_path", None)

        if file_path:
            return str(file_path)

        result_id = getattr(result, "id", None)

        if result_id:
            return str(result_id)

        raise TypeError(
            "Retriever result must be a string, dictionary, "
            "or object containing file_path/id"
        )

    def _normalise_results(
        self,
        results: Sequence[Any] | Iterable[Any],
    ) -> list[str]:
        """Convert retriever results into evaluation identifiers."""

        return [
            self._result_identifier(result)
            for result in results
        ]

    def evaluate(
        self,
        name: str,
        retriever: Callable[[EvaluationExample], Sequence[Any]],
    ) -> BenchmarkResult:
        """Evaluate a single retrieval configuration."""

        if not name.strip():
            raise ValueError("name must not be empty")

        rankings: list[list[str]] = []
        relevant_sets: list[set[str]] = []

        hit_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        ndcg_scores: list[float] = []

        for example in self.dataset:
            raw_results = retriever(example)

            retrieved = self._normalise_results(raw_results)
            relevant = self._relevant_ids(example)

            rankings.append(retrieved)
            relevant_sets.append(relevant)

            hit_scores.append(
                hit_rate_at_k(
                    retrieved,
                    relevant,
                    self.top_k,
                )
            )

            precision_scores.append(
                precision_at_k(
                    retrieved,
                    relevant,
                    self.top_k,
                )
            )

            recall_scores.append(
                recall_at_k(
                    retrieved,
                    relevant,
                    self.top_k,
                )
            )

            ndcg_scores.append(
                ndcg_at_k(
                    retrieved,
                    relevant,
                    self.top_k,
                )
            )

        if not self.dataset:
            return BenchmarkResult(
                name=name,
                hit_rate=0.0,
                precision=0.0,
                recall=0.0,
                mrr=0.0,
                ndcg=0.0,
            )

        return BenchmarkResult(
            name=name,
            hit_rate=sum(hit_scores) / len(hit_scores),
            precision=sum(precision_scores) / len(precision_scores),
            recall=sum(recall_scores) / len(recall_scores),
            mrr=mean_reciprocal_rank(
                rankings,
                relevant_sets,
            ),
            ndcg=sum(ndcg_scores) / len(ndcg_scores),
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
            self.evaluate(
                name,
                retriever,
            )
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
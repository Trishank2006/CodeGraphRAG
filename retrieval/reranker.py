from __future__ import annotations

from dataclasses import replace
from typing import Any

from retrieval.models import RetrievalResult


class CrossEncoderReranker:
    """
    Rerank retrieval candidates using a cross-encoder model.

    The model is loaded lazily on the first rerank call. A model object
    can also be injected directly, which keeps unit tests independent
    of model downloads.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: Any | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size

    def _get_model(self) -> Any:
        """Load the cross-encoder lazily."""
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)

        return self.model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Rerank candidate results for a query.

        Only the supplied candidate list is reranked. This is intended
        to be called after hybrid/RRF retrieval has already reduced the
        search space.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        if not query.strip() or not results or top_k == 0:
            return []

        model = self._get_model()

        pairs = [
            (query, result.content)
            for result in results
        ]

        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
        )

        if len(scores) != len(results):
            raise ValueError(
                "Reranker returned a score count that does not "
                "match the candidate count"
            )

        reranked = [
            replace(
                result,
                score=float(score),
            )
            for result, score in zip(results, scores)
        ]

        # Deterministic ordering for equal model scores.
        reranked.sort(
            key=lambda result: (-result.score, result.id)
        )

        return reranked[:top_k]


def rerank_results(
    query: str,
    results: list[RetrievalResult],
    top_k: int = 10,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    model: Any | None = None,
    batch_size: int = 32,
) -> list[RetrievalResult]:
    """Convenience function for cross-encoder reranking."""
    reranker = CrossEncoderReranker(
        model_name=model_name,
        model=model,
        batch_size=batch_size,
    )

    return reranker.rerank(
        query=query,
        results=results,
        top_k=top_k,
    )
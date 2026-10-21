from __future__ import annotations

from dataclasses import replace
from typing import Any

from retrieval.models import RetrievalResult
from retrieval.reranker import CrossEncoderReranker


class MetadataAwareReranker(CrossEncoderReranker):
    """Experimental reranker that includes code metadata in its input."""

    @staticmethod
    def _format_candidate(result: RetrievalResult) -> str:
        return (
            f"File path: {result.file_path}\n"
            f"Symbol: {result.symbol}\n"
            f"Code:\n{result.content}"
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        if not query.strip() or not results or top_k == 0:
            return []

        model = self._get_model()
        pairs = [
            (query, self._format_candidate(result))
            for result in results
        ]

        scores = model.predict(pairs, batch_size=self.batch_size)

        if len(scores) != len(results):
            raise ValueError(
                "Reranker returned a score count that does not "
                "match the candidate count"
            )

        reranked = [
            replace(result, score=float(score))
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda result: (-result.score, result.id))
        return reranked[:top_k]

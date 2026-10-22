
from __future__ import annotations

from dataclasses import replace

from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.hybrid import DenseSearch, GraphSearch, HybridRetriever
from retrieval.models import RetrievalResult
from retrieval.reranker import CrossEncoderReranker



def weighted_rank_fusion(
    hybrid_results: list[RetrievalResult],
    reranked_results: list[RetrievalResult],
    weight: float,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse hybrid and cross-encoder rankings using reciprocal ranks."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")

    hybrid_rank = {
        result.id: rank
        for rank, result in enumerate(hybrid_results, start=1)
        if result.id
    }
    reranker_rank = {
        result.id: rank
        for rank, result in enumerate(reranked_results, start=1)
        if result.id
    }

    # Retain the union of both candidate sets.
    # Prefer reranker metadata for candidates it returned.
    results_by_id = {
        result.id: result
        for result in hybrid_results
        if result.id
    }
    results_by_id.update({
        result.id: result
        for result in reranked_results
        if result.id
    })

    fused = []
    for result_id, result in results_by_id.items():
        hybrid_position = hybrid_rank.get(result_id)
        cross_position = reranker_rank.get(result_id)

        score = 0.0
        if hybrid_position is not None:
            score += (1.0 - weight) / (k + hybrid_position)
        if cross_position is not None:
            score += weight / (k + cross_position)

        fused.append(replace(result, score=score))

    fused.sort(key=lambda result: (-result.score, result.id))

    if top_k is not None:
        fused = fused[:top_k]

    return fused



class HybridRetrievalPipeline:
    """
    End-to-end retrieval pipeline.

    Dense + BM25 + Graph -> RRF -> Cross-Encoder -> optional rank fusion.
    """

    def __init__(
        self,
        dense_search: DenseSearch,
        bm25_retriever: BM25Retriever,
        graph_search: GraphSearch | None = None,
        reranker: CrossEncoderReranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()

        self.hybrid = HybridRetriever(
            dense_search=dense_search,
            bm25_retriever=bm25_retriever,
            graph_search=graph_search,
            rrf_k=self.config.rrf_k,
        )

        self.reranker = reranker

    def search(self, query: str) -> list[RetrievalResult]:
        """Run hybrid retrieval followed by optional reranking."""
        fused_results = self.hybrid.search(
            query=query,
            dense_top_k=self.config.dense_top_k,
            bm25_top_k=self.config.bm25_top_k,
            graph_top_k=self.config.graph_top_k,
            rrf_top_k=self.config.rrf_top_k,
        )

        if self.reranker is None:
            return fused_results

        # Preserve the original reranking behavior unless rank fusion
        # is explicitly enabled.
        if self.config.fusion_weight is None:
            return self.reranker.rerank(
                query=query,
                results=fused_results,
                top_k=self.config.rerank_top_k,
            )

        # Rerank the entire candidate pool before applying the final limit.
        reranked_results = self.reranker.rerank(
            query=query,
            results=fused_results,
            top_k=len(fused_results),
        )

        return weighted_rank_fusion(
            hybrid_results=fused_results,
            reranked_results=reranked_results,
            weight=self.config.fusion_weight,
            k=self.config.rrf_k,
            top_k=self.config.rerank_top_k,
        )

from __future__ import annotations

from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.hybrid import DenseSearch, GraphSearch, HybridRetriever
from retrieval.models import RetrievalResult
from retrieval.reranker import CrossEncoderReranker


class HybridRetrievalPipeline:
    """
    End-to-end Phase 4 retrieval pipeline.

    Flow:
        Dense + BM25 + Graph
                ↓
              RRF
                ↓
        Cross-Encoder Reranker
                ↓
          final results
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
        """
        Run hybrid retrieval followed by optional reranking.
        """
        fused_results = self.hybrid.search(
            query=query,
            dense_top_k=self.config.dense_top_k,
            bm25_top_k=self.config.bm25_top_k,
            graph_top_k=self.config.graph_top_k,
            rrf_top_k=self.config.rrf_top_k,
        )

        if self.reranker is None:
            return fused_results

        return self.reranker.rerank(
            query=query,
            results=fused_results,
            top_k=self.config.rerank_top_k,
        )
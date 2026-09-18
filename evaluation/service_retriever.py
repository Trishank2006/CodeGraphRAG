from __future__ import annotations

from typing import Callable, Dict, List

from application.service import CodeGraphRAGService
from retrieval.models import RetrievalResult
from retrieval.search import search_code
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.config import RetrievalConfig
from graph.retrieval import GraphRetriever


class ServiceRetriever:
    """
    Adapter between CodeGraphRAGService and the Phase 7
    evaluation framework.

    Provides independent retrieval configurations:

    1. Dense
    2. BM25
    3. Dense + BM25
    4. Dense + BM25 + Graph
    5. Hybrid + RRF
    6. Hybrid + RRF + Reranker
    """

    def __init__(
        self,
        service: CodeGraphRAGService,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.service = service
        self.config = config or RetrievalConfig()

    def dense(self, query: str) -> List[RetrievalResult]:
        """Dense vector retrieval only."""
        return search_code(
            query=query,
            top_k=self.config.dense_top_k,
            embedder=self.service.embedder,
            store=self.service.vector_store,
        )

    def bm25(self, query: str) -> List[RetrievalResult]:
        """BM25 retrieval only."""
        return self.service.bm25_retriever.search_bm25(
            query,
            top_k=self.config.bm25_top_k,
        )

    def dense_bm25(self, query: str) -> List[RetrievalResult]:
        """
        Dense + BM25 retrieval using RRF fusion.

        Graph and reranking are disabled.
        """
        pipeline = HybridRetrievalPipeline(
            dense_search=self._dense_search,
            bm25_retriever=self.service.bm25_retriever,
            graph_search=None,
            reranker=None,
            config=RetrievalConfig(
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                graph_top_k=0,
                rrf_top_k=self.config.rrf_top_k,
                rerank_top_k=self.config.rerank_top_k,
                rrf_k=self.config.rrf_k,
            ),
        )

        return pipeline.search(query)

    def dense_bm25_graph(self, query: str) -> List[RetrievalResult]:
        """
        Dense + BM25 + Graph retrieval.

        Reranking is disabled.
        """
        graph_retriever = GraphRetriever(self.service.graph_store)

        pipeline = HybridRetrievalPipeline(
            dense_search=self._dense_search,
            bm25_retriever=self.service.bm25_retriever,
            graph_search=lambda q, k: graph_retriever.search(
                q,
                top_k=k,
            ),
            reranker=None,
            config=RetrievalConfig(
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                graph_top_k=self.config.graph_top_k,
                rrf_top_k=self.config.rrf_top_k,
                rerank_top_k=self.config.rerank_top_k,
                rrf_k=self.config.rrf_k,
            ),
        )

        return pipeline.search(query)

    def hybrid_rrf(self, query: str) -> List[RetrievalResult]:
        """
        Hybrid Dense + BM25 + RRF retrieval.

        Graph and reranking are disabled.

        Note:
        RRF is the fusion mechanism used by the hybrid pipeline.
        """
        pipeline = HybridRetrievalPipeline(
            dense_search=self._dense_search,
            bm25_retriever=self.service.bm25_retriever,
            graph_search=None,
            reranker=None,
            config=RetrievalConfig(
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                graph_top_k=0,
                rrf_top_k=self.config.rrf_top_k,
                rerank_top_k=self.config.rerank_top_k,
                rrf_k=self.config.rrf_k,
            ),
        )

        return pipeline.search(query)

    def hybrid_rrf_reranker(
        self,
        query: str,
    ) -> List[RetrievalResult]:
        """
        Full retrieval pipeline:

        Dense + BM25 + Graph + RRF + Reranker.
        """
        graph_retriever = GraphRetriever(self.service.graph_store)

        pipeline = HybridRetrievalPipeline(
            dense_search=self._dense_search,
            bm25_retriever=self.service.bm25_retriever,
            graph_search=lambda q, k: graph_retriever.search(
                q,
                top_k=k,
            ),
            reranker=self.service.reranker,
            config=RetrievalConfig(
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                graph_top_k=self.config.graph_top_k,
                rrf_top_k=self.config.rrf_top_k,
                rerank_top_k=self.config.rerank_top_k,
                rrf_k=self.config.rrf_k,
            ),
        )

        return pipeline.search(query)

    def _dense_search(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Internal dense retrieval callback."""
        return search_code(
            query=query,
            top_k=top_k,
            embedder=self.service.embedder,
            store=self.service.vector_store,
        )

    def as_retrievers(
        self,
    ) -> Dict[str, Callable]:
        """
        Return benchmark-ready retrieval functions.
        """
        return {
            "Dense": lambda example: self.dense(
                example.query
            ),
            "BM25": lambda example: self.bm25(
                example.query
            ),
            "Dense + BM25": lambda example: self.dense_bm25(
                example.query
            ),
            "Dense + BM25 + Graph": lambda example: (
                self.dense_bm25_graph(example.query)
            ),
            "Hybrid + RRF": lambda example: self.hybrid_rrf(
                example.query
            ),
            "Hybrid + RRF + Reranker": lambda example: (
                self.hybrid_rrf_reranker(example.query)
            ),
        }
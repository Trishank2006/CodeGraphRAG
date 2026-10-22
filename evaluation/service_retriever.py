from __future__ import annotations

from typing import Callable

from application.service import CodeGraphRAGService
from graph.retrieval import GraphRetriever
from retrieval.config import RetrievalConfig
from retrieval.models import RetrievalResult
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.search import search_code


class ServiceRetriever:
    """Adapter between CodeGraphRAGService and benchmark configurations."""

    def __init__(
        self,
        service: CodeGraphRAGService,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.service = service
        self.config = config or RetrievalConfig()

    def dense(self, query: str) -> list[RetrievalResult]:
        """Dense vector retrieval only."""
        return self._dense_search(query, self.config.dense_top_k)

    def bm25(self, query: str) -> list[RetrievalResult]:
        """BM25 retrieval only."""
        return self.service.bm25_retriever.search_bm25(
            query,
            top_k=self.config.bm25_top_k,
        )

    def _dense_search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Internal dense retrieval callback."""
        return search_code(
            query=query,
            top_k=top_k,
            embedder=self.service.embedder,
            store=self.service.vector_store,
        )

    def _hybrid_pipeline(
        self,
        *,
        include_graph: bool,
        include_reranker: bool,
    ) -> HybridRetrievalPipeline:
        """Build a hybrid pipeline with selected components."""
        graph_search = None

        if include_graph:
            graph_retriever = GraphRetriever(self.service.graph_store)

            graph_search = lambda query, top_k: graph_retriever.search(
                query,
                top_k=top_k,
            )

        return HybridRetrievalPipeline(
            dense_search=self._dense_search,
            bm25_retriever=self.service.bm25_retriever,
            graph_search=graph_search,
            reranker=(
                self.service.reranker if include_reranker else None
            ),
            config=RetrievalConfig(
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                graph_top_k=(
                    self.config.graph_top_k if include_graph else 0
                ),
                rrf_top_k=self.config.rrf_top_k,
                rerank_top_k=self.config.rerank_top_k,
                rrf_k=self.config.rrf_k,
                fusion_weight=self.config.fusion_weight,
            ),
        )

    def dense_bm25(self, query: str) -> list[RetrievalResult]:
        """Dense + BM25 using RRF, without graph or reranking."""
        return self._hybrid_pipeline(
            include_graph=False,
            include_reranker=False,
        ).search(query)

    def dense_bm25_graph(self, query: str) -> list[RetrievalResult]:
        """Dense + BM25 + graph, without reranking."""
        return self._hybrid_pipeline(
            include_graph=True,
            include_reranker=False,
        ).search(query)

    def dense_bm25_graph_reranker(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        """Dense + BM25 + graph + reranking."""
        return self._hybrid_pipeline(
            include_graph=True,
            include_reranker=True,
        ).search(query)

    def as_retrievers(self) -> dict[str, Callable]:
        """Return distinct benchmark configurations."""
        return {
            "Dense": lambda example: self.dense(example.query),
            "BM25": lambda example: self.bm25(example.query),
            "Dense + BM25 (RRF)": lambda example: self.dense_bm25(
                example.query
            ),
            "Dense + BM25 + Graph": lambda example: self.dense_bm25_graph(
                example.query
            ),
            "Dense + BM25 + Graph + Reranker": (
                lambda example: self.dense_bm25_graph_reranker(
                    example.query
                )
            ),
        }
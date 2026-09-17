from __future__ import annotations

from collections.abc import Callable

from retrieval.bm25 import BM25Retriever
from retrieval.models import RetrievalResult
from retrieval.rrf import reciprocal_rank_fusion


DenseSearch = Callable[[str, int], list[RetrievalResult]]
GraphSearch = Callable[[str, int], list[RetrievalResult]]


class HybridRetriever:
    """
    Coordinates dense, BM25, and graph retrieval.

    Each retriever produces RetrievalResult objects. The results are
    combined with Reciprocal Rank Fusion rather than adding raw scores.
    """

    def __init__(
        self,
        dense_search: DenseSearch,
        bm25_retriever: BM25Retriever,
        graph_search: GraphSearch | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense_search = dense_search
        self.bm25_retriever = bm25_retriever
        self.graph_search = graph_search
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        graph_top_k: int = 20,
        rrf_top_k: int = 30,
    ) -> list[RetrievalResult]:
        """
        Execute all configured retrieval methods and fuse their results.

        Graph retrieval is optional. If no graph_search callable was
        supplied, hybrid retrieval uses dense + BM25 only.
        """
        if not query or not query.strip():
            return []

        if dense_top_k < 0:
            raise ValueError("dense_top_k must be non-negative")

        if bm25_top_k < 0:
            raise ValueError("bm25_top_k must be non-negative")

        if graph_top_k < 0:
            raise ValueError("graph_top_k must be non-negative")

        if rrf_top_k < 0:
            raise ValueError("rrf_top_k must be non-negative")

        result_lists: list[list[RetrievalResult]] = []

        dense_results = self.dense_search(
            query,
            dense_top_k,
        )
        result_lists.append(dense_results)

        bm25_results = self.bm25_retriever.search_bm25(
            query,
            top_k=bm25_top_k,
        )
        result_lists.append(bm25_results)

        if self.graph_search is not None and graph_top_k > 0:
            graph_results = self.graph_search(
                query,
                graph_top_k,
            )
            result_lists.append(graph_results)

        return reciprocal_rank_fusion(
            result_lists,
            k=self.rrf_k,
            top_k=rrf_top_k,
        )


def create_hybrid_retriever(
    dense_search: DenseSearch,
    bm25_retriever: BM25Retriever,
    graph_search: GraphSearch | None = None,
    rrf_k: int = 60,
) -> HybridRetriever:
    """Create a configured HybridRetriever instance."""
    return HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=bm25_retriever,
        graph_search=graph_search,
        rrf_k=rrf_k,
    )
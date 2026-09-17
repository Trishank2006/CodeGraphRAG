from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.models import RetrievalResult


def make_result(
    result_id: str,
    source: str,
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        source=source,
        score=1.0,
        repository="test-repo",
        file_path=f"src/{result_id}.py",
        language="python",
        symbol=result_id,
        start_line=1,
        end_line=10,
        content=f"content for {result_id}",
    )


def dense_search(
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    del query
    return [
        make_result("a", "dense"),
        make_result("shared", "dense"),
    ][:top_k]


def graph_search(
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    del query
    return [
        make_result("shared", "graph"),
        make_result("c", "graph"),
    ][:top_k]


def make_bm25() -> BM25Retriever:
    retriever = BM25Retriever()

    retriever.index(
        [
            {
                "chunk_id": "shared",
                "repository": "test-repo",
                "file_path": "src/shared.py",
                "language": "python",
                "symbol": "shared",
                "start_line": 1,
                "end_line": 10,
                "content": "shared implementation",
            },
            {
                "chunk_id": "b",
                "repository": "test-repo",
                "file_path": "src/b.py",
                "language": "python",
                "symbol": "b",
                "start_line": 1,
                "end_line": 10,
                "content": "payment implementation",
            },
        ]
    )

    return retriever


def test_hybrid_combines_dense_bm25_and_graph():
    retriever = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    results = retriever.search(
        "shared",
        dense_top_k=2,
        bm25_top_k=2,
        graph_top_k=2,
        rrf_top_k=10,
    )

    ids = [result.id for result in results]

    assert "shared" in ids
    assert "a" in ids
    assert "c" in ids


def test_hybrid_deduplicates_results():
    retriever = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    results = retriever.search(
        "shared",
        dense_top_k=2,
        bm25_top_k=2,
        graph_top_k=2,
        rrf_top_k=10,
    )

    ids = [result.id for result in results]

    assert ids.count("shared") == 1


def test_hybrid_preserves_fused_source_information():
    retriever = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    results = retriever.search(
        "shared",
        dense_top_k=2,
        bm25_top_k=2,
        graph_top_k=2,
        rrf_top_k=10,
    )

    shared = next(result for result in results if result.id == "shared")

    assert "dense" in shared.source
    assert "graph" in shared.source


def test_hybrid_respects_rrf_top_k():
    retriever = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    results = retriever.search(
        "shared",
        dense_top_k=2,
        bm25_top_k=2,
        graph_top_k=2,
        rrf_top_k=2,
    )

    assert len(results) == 2


def test_hybrid_works_without_graph():
    retriever = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
    )

    results = retriever.search(
        "shared",
        dense_top_k=2,
        bm25_top_k=2,
        graph_top_k=2,
        rrf_top_k=10,
    )

    assert len(results) > 0


def test_hybrid_empty_query_returns_empty():
    retriever = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
    )

    assert retriever.search("   ") == []


def test_hybrid_rejects_negative_top_k():
    retriever = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
    )

    try:
        retriever.search("shared", dense_top_k=-1)
        assert False
    except ValueError:
        pass
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.models import RetrievalResult
from retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, batch_size=32):
        assert len(pairs) == len(self.scores)
        return self.scores


def make_result(
    result_id: str,
    source: str,
    content: str,
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
        content=content,
    )


def dense_search(
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    del query

    results = [
        make_result("auth", "dense", "authentication logic"),
        make_result("payment", "dense", "payment processing"),
        make_result("logging", "dense", "logging utilities"),
    ]

    return results[:top_k]


def graph_search(
    query: str,
    top_k: int,
) -> list[RetrievalResult]:
    del query

    results = [
        make_result("auth", "graph", "authentication logic"),
        make_result("database", "graph", "database access"),
        make_result("logging", "graph", "logging utilities"),
    ]

    return results[:top_k]


def make_bm25() -> BM25Retriever:
    retriever = BM25Retriever()

    retriever.index(
        [
            {
                "chunk_id": "auth",
                "repository": "test-repo",
                "file_path": "src/auth.py",
                "language": "python",
                "symbol": "authenticate_user",
                "start_line": 1,
                "end_line": 10,
                "content": "authentication user login password verification",
            },
            {
                "chunk_id": "payment",
                "repository": "test-repo",
                "file_path": "src/payment.py",
                "language": "python",
                "symbol": "process_payment",
                "start_line": 1,
                "end_line": 10,
                "content": "payment processing transaction",
            },
            {
                "chunk_id": "logging",
                "repository": "test-repo",
                "file_path": "src/logging.py",
                "language": "python",
                "symbol": "write_log",
                "start_line": 1,
                "end_line": 10,
                "content": "logging utilities debug messages",
            },
        ]
    )

    return retriever


def test_hybrid_end_to_end_dense_bm25_graph_and_reranker():
    hybrid = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    fused_results = hybrid.search(
        "authentication",
        dense_top_k=3,
        bm25_top_k=3,
        graph_top_k=3,
        rrf_top_k=3,
    )

    assert len(fused_results) == 3
    assert all(
        isinstance(result, RetrievalResult)
        for result in fused_results
    )

    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder([0.2, 0.95, 0.5])
    )

    final_results = reranker.rerank(
        "authentication",
        fused_results,
        top_k=2,
    )

    assert len(final_results) == 2
    assert final_results[0].score == 0.95
    assert final_results[1].score == 0.5


def test_hybrid_end_to_end_deduplicates_before_reranking():
    hybrid = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    fused_results = hybrid.search(
        "authentication",
        dense_top_k=3,
        bm25_top_k=3,
        graph_top_k=3,
        rrf_top_k=10,
    )

    ids = [result.id for result in fused_results]

    assert len(ids) == len(set(ids))


def test_hybrid_end_to_end_preserves_retrieval_metadata():
    hybrid = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    fused_results = hybrid.search(
        "authentication",
        dense_top_k=3,
        bm25_top_k=3,
        graph_top_k=3,
        rrf_top_k=10,
    )

    for result in fused_results:
        assert result.repository == "test-repo"
        assert result.file_path.startswith("src/")
        assert result.language == "python"
        assert result.content


def test_hybrid_end_to_end_reranker_limits_final_results():
    hybrid = HybridRetriever(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    fused_results = hybrid.search(
        "authentication",
        dense_top_k=3,
        bm25_top_k=3,
        graph_top_k=3,
        rrf_top_k=3,
    )

    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder([0.8, 0.7, 0.9])
    )

    final_results = reranker.rerank(
        "authentication",
        fused_results,
        top_k=1,
    )

    assert len(final_results) == 1
    assert final_results[0].score == 0.9
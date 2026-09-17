from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.models import RetrievalResult
from retrieval.pipeline import HybridRetrievalPipeline
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


def test_pipeline_returns_fused_results_without_reranker():
    pipeline = HybridRetrievalPipeline(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
    )

    results = pipeline.search("authentication")

    assert results
    assert all(
        isinstance(result, RetrievalResult)
        for result in results
    )


def test_pipeline_applies_reranker():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder([0.2, 0.9, 0.5])
    )

    config = RetrievalConfig(
        dense_top_k=3,
        bm25_top_k=3,
        graph_top_k=3,
        rrf_top_k=3,
        rerank_top_k=2,
        rrf_k=60,
    )

    pipeline = HybridRetrievalPipeline(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
        reranker=reranker,
        config=config,
    )

    results = pipeline.search("authentication")

    assert len(results) == 2
    assert results[0].score == 0.9
    assert results[1].score == 0.5


def test_pipeline_uses_config_limits():
    config = RetrievalConfig(
        dense_top_k=1,
        bm25_top_k=1,
        graph_top_k=1,
        rrf_top_k=1,
        rerank_top_k=1,
        rrf_k=60,
    )

    pipeline = HybridRetrievalPipeline(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
        config=config,
    )

    results = pipeline.search("authentication")

    assert len(results) == 1


def test_pipeline_works_without_graph_or_reranker():
    pipeline = HybridRetrievalPipeline(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
    )

    results = pipeline.search("authentication")

    assert results
    assert all(
        isinstance(result, RetrievalResult)
        for result in results
    )


def test_pipeline_respects_custom_rrf_k():
    config = RetrievalConfig(
        dense_top_k=3,
        bm25_top_k=3,
        graph_top_k=3,
        rrf_top_k=3,
        rerank_top_k=3,
        rrf_k=10,
    )

    pipeline = HybridRetrievalPipeline(
        dense_search=dense_search,
        bm25_retriever=make_bm25(),
        graph_search=graph_search,
        config=config,
    )

    assert pipeline.hybrid.rrf_k == 10
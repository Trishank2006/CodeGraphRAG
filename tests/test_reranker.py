from retrieval.models import RetrievalResult
from retrieval.reranker import CrossEncoderReranker, rerank_results


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.received_pairs = None
        self.received_batch_size = None

    def predict(self, pairs, batch_size=32):
        self.received_pairs = pairs
        self.received_batch_size = batch_size
        return self.scores


def make_result(
    result_id: str,
    content: str,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        source="hybrid",
        score=score,
        repository="test-repo",
        file_path=f"src/{result_id}.py",
        language="python",
        symbol=result_id,
        start_line=1,
        end_line=10,
        content=content,
    )


def test_reranker_sorts_by_cross_encoder_score():
    results = [
        make_result("a", "first result"),
        make_result("b", "second result"),
        make_result("c", "third result"),
    ]

    model = FakeCrossEncoder([0.2, 0.9, 0.5])

    reranker = CrossEncoderReranker(
        model=model,
        batch_size=16,
    )

    reranked = reranker.rerank(
        "test query",
        results,
        top_k=3,
    )

    assert [result.id for result in reranked] == ["b", "c", "a"]
    assert [result.score for result in reranked] == [0.9, 0.5, 0.2]


def test_reranker_respects_top_k():
    results = [
        make_result("a", "first result"),
        make_result("b", "second result"),
        make_result("c", "third result"),
    ]

    model = FakeCrossEncoder([0.2, 0.9, 0.5])

    reranker = CrossEncoderReranker(model=model)

    reranked = reranker.rerank(
        "test query",
        results,
        top_k=2,
    )

    assert len(reranked) == 2
    assert [result.id for result in reranked] == ["b", "c"]


def test_reranker_passes_query_and_content_pairs():
    results = [
        make_result("a", "first result"),
        make_result("b", "second result"),
    ]

    model = FakeCrossEncoder([0.8, 0.6])

    reranker = CrossEncoderReranker(
        model=model,
        batch_size=8,
    )

    reranker.rerank(
        "find authentication",
        results,
        top_k=2,
    )

    assert model.received_pairs == [
        ("find authentication", "first result"),
        ("find authentication", "second result"),
    ]
    assert model.received_batch_size == 8


def test_reranker_is_deterministic_for_equal_scores():
    results = [
        make_result("b", "result b"),
        make_result("a", "result a"),
    ]

    model = FakeCrossEncoder([0.5, 0.5])

    reranker = CrossEncoderReranker(model=model)

    reranked = reranker.rerank(
        "test query",
        results,
        top_k=2,
    )

    assert [result.id for result in reranked] == ["a", "b"]


def test_reranker_empty_query_returns_empty():
    results = [
        make_result("a", "result"),
    ]

    model = FakeCrossEncoder([0.9])

    reranker = CrossEncoderReranker(model=model)

    assert reranker.rerank("   ", results) == []


def test_reranker_empty_results_returns_empty():
    model = FakeCrossEncoder([])

    reranker = CrossEncoderReranker(model=model)

    assert reranker.rerank("test query", []) == []


def test_reranker_zero_top_k_returns_empty():
    results = [
        make_result("a", "result"),
    ]

    model = FakeCrossEncoder([0.9])

    reranker = CrossEncoderReranker(model=model)

    assert reranker.rerank(
        "test query",
        results,
        top_k=0,
    ) == []


def test_reranker_rejects_negative_top_k():
    model = FakeCrossEncoder([])

    reranker = CrossEncoderReranker(model=model)

    try:
        reranker.rerank(
            "test query",
            [],
            top_k=-1,
        )
        assert False
    except ValueError:
        pass


def test_reranker_rejects_invalid_batch_size():
    try:
        CrossEncoderReranker(batch_size=0)
        assert False
    except ValueError:
        pass


def test_reranker_rejects_score_count_mismatch():
    results = [
        make_result("a", "result"),
        make_result("b", "result"),
    ]

    model = FakeCrossEncoder([0.9])

    reranker = CrossEncoderReranker(model=model)

    try:
        reranker.rerank(
            "test query",
            results,
            top_k=2,
        )
        assert False
    except ValueError:
        pass


def test_rerank_results_convenience_function():
    results = [
        make_result("a", "result a"),
        make_result("b", "result b"),
    ]

    model = FakeCrossEncoder([0.3, 0.8])

    reranked = rerank_results(
        query="test query",
        results=results,
        top_k=2,
        model=model,
    )

    assert [result.id for result in reranked] == ["b", "a"]
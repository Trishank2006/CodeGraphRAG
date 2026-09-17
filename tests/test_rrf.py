from retrieval.models import RetrievalResult
from retrieval.rrf import reciprocal_rank_fusion


def make_result(result_id: str, source: str, score: float = 1.0):
    return RetrievalResult(
        id=result_id,
        source=source,
        score=score,
        repository="test-repo",
        file_path=f"src/{result_id}.py",
        language="python",
        symbol=result_id,
        start_line=1,
        end_line=10,
        content=f"content for {result_id}",
    )


def test_rrf_combines_rank_scores():
    dense = [
        make_result("a", "dense"),
        make_result("b", "dense"),
    ]

    bm25 = [
        make_result("b", "bm25"),
        make_result("c", "bm25"),
    ]

    results = reciprocal_rank_fusion([dense, bm25], k=60)

    assert [result.id for result in results] == ["b", "a", "c"]

    assert results[0].score == (
        1 / 61 + 1 / 62
    )


def test_rrf_deduplicates_results_and_preserves_sources():
    dense = [
        make_result("shared", "dense"),
    ]

    bm25 = [
        make_result("shared", "bm25"),
    ]

    results = reciprocal_rank_fusion([dense, bm25], k=60)

    assert len(results) == 1
    assert results[0].id == "shared"
    assert results[0].source == "dense+bm25"


def test_rrf_respects_top_k():
    results_a = [
        make_result("a", "dense"),
        make_result("b", "dense"),
        make_result("c", "dense"),
    ]

    results = reciprocal_rank_fusion(
        [results_a],
        k=60,
        top_k=2,
    )

    assert len(results) == 2
    assert [result.id for result in results] == ["a", "b"]


def test_rrf_is_deterministic_for_equal_scores():
    dense = [
        make_result("b", "dense"),
    ]

    bm25 = [
        make_result("a", "bm25"),
    ]

    results = reciprocal_rank_fusion([dense, bm25], k=60)

    assert [result.id for result in results] == ["a", "b"]


def test_rrf_rejects_invalid_k():
    try:
        reciprocal_rank_fusion([], k=-1)
        assert False
    except ValueError:
        pass


def test_rrf_rejects_invalid_top_k():
    try:
        reciprocal_rank_fusion([], top_k=-1)
        assert False
    except ValueError:
        pass

import pytest

from retrieval.models import RetrievalResult
from retrieval.pipeline import weighted_rank_fusion


def make_result(result_id: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        source="test",
        score=score,
        repository="test-repo",
        file_path=f"{result_id}.py",
        language="python",
        symbol=result_id,
        start_line=1,
        end_line=10,
        content=f"Content for {result_id}",
    )


def test_weighted_rank_fusion_uses_both_rankings():
    hybrid = [make_result("A"), make_result("B"), make_result("C")]
    reranked = [make_result("C"), make_result("B"), make_result("A")]

    results = weighted_rank_fusion(hybrid, reranked, weight=0.5)

    assert [result.id for result in results] == ["A", "C", "B"]


def test_weight_one_preserves_reranker_order():
    hybrid = [make_result("A"), make_result("B"), make_result("C")]
    reranked = [make_result("C"), make_result("B"), make_result("A")]

    results = weighted_rank_fusion(hybrid, reranked, weight=1.0)

    assert [result.id for result in results] == ["C", "B", "A"]


def test_weight_zero_uses_hybrid_rank_for_reranked_candidates():
    hybrid = [make_result("A"), make_result("B"), make_result("C")]
    reranked = [make_result("C"), make_result("B"), make_result("A")]

    results = weighted_rank_fusion(hybrid, reranked, weight=0.0)

    assert [result.id for result in results] == ["A", "B", "C"]


def test_weighted_rank_fusion_respects_top_k():
    hybrid = [make_result("A"), make_result("B"), make_result("C")]
    reranked = [make_result("C"), make_result("B"), make_result("A")]

    results = weighted_rank_fusion(
        hybrid_results=hybrid,
        reranked_results=reranked,
        weight=0.5,
        top_k=2,
    )

    assert len(results) == 2


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_weighted_rank_fusion_rejects_invalid_weight(weight):
    with pytest.raises(ValueError, match="weight"):
        weighted_rank_fusion([], [], weight=weight)


def test_weighted_rank_fusion_rejects_negative_k():
    with pytest.raises(ValueError, match="k"):
        weighted_rank_fusion([], [], weight=0.5, k=-1)


def test_weighted_rank_fusion_rejects_negative_top_k():
    with pytest.raises(ValueError, match="top_k"):
        weighted_rank_fusion([], [], weight=0.5, top_k=-1)


from unittest.mock import Mock

from retrieval.pipeline import HybridRetrievalPipeline


def make_pipeline(config, hybrid_results, reranked_results=None):
    bm25 = Mock()
    reranker = Mock()

    pipeline = HybridRetrievalPipeline(
        dense_search=Mock(return_value=[]),
        bm25_retriever=bm25,
        graph_search=Mock(return_value=[]),
        reranker=reranker,
        config=config,
    )

    pipeline.hybrid = Mock()
    pipeline.hybrid.search.return_value = hybrid_results

    if reranked_results is not None:
        reranker.rerank.return_value = reranked_results

    return pipeline, reranker


def test_pipeline_preserves_default_reranker_behavior():
    from retrieval.config import RetrievalConfig

    candidates = [make_result("A"), make_result("B")]
    reranked = [make_result("B"), make_result("A")]
    config = RetrievalConfig(rerank_top_k=1)

    pipeline, reranker = make_pipeline(config, candidates, reranked)

    results = pipeline.search("test query")

    reranker.rerank.assert_called_once_with(
        query="test query",
        results=candidates,
        top_k=1,
    )
    assert results == reranked


def test_pipeline_applies_weighted_fusion_when_enabled():
    from retrieval.config import RetrievalConfig

    candidates = [make_result("A"), make_result("B"), make_result("C")]
    reranked = [make_result("C"), make_result("B"), make_result("A")]
    config = RetrievalConfig(rerank_top_k=2, fusion_weight=0.5)

    pipeline, reranker = make_pipeline(config, candidates, reranked)

    results = pipeline.search("test query")

    reranker.rerank.assert_called_once_with(
        query="test query",
        results=candidates,
        top_k=len(candidates),
    )
    assert len(results) == 2
    assert [result.id for result in results] == ["A", "C"]


def test_pipeline_without_reranker_returns_hybrid_results():
    from retrieval.config import RetrievalConfig

    candidates = [make_result("A"), make_result("B")]

    pipeline = HybridRetrievalPipeline(
        dense_search=Mock(return_value=[]),
        bm25_retriever=Mock(),
        graph_search=Mock(return_value=[]),
        reranker=None,
        config=RetrievalConfig(fusion_weight=0.5),
    )
    pipeline.hybrid = Mock()
    pipeline.hybrid.search.return_value = candidates

    assert pipeline.search("test query") == candidates


def test_weighted_rank_fusion_preserves_hybrid_only_candidates():
    hybrid = [
        make_result("A"),
        make_result("B"),
        make_result("C"),
    ]
    reranked = [
        make_result("C"),
        make_result("A"),
    ]

    results = weighted_rank_fusion(
        hybrid_results=hybrid,
        reranked_results=reranked,
        weight=0.5,
        top_k=3,
    )

    assert {result.id for result in results} == {"A", "B", "C"}
    assert len(results) == 3


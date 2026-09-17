import pytest

from retrieval.config import RetrievalConfig


def test_default_retrieval_config():
    config = RetrievalConfig()

    assert config.dense_top_k == 20
    assert config.bm25_top_k == 20
    assert config.graph_top_k == 20
    assert config.rrf_top_k == 30
    assert config.rerank_top_k == 10
    assert config.rrf_k == 60


def test_custom_retrieval_config():
    config = RetrievalConfig(
        dense_top_k=5,
        bm25_top_k=7,
        graph_top_k=9,
        rrf_top_k=12,
        rerank_top_k=4,
        rrf_k=50,
    )

    assert config.dense_top_k == 5
    assert config.bm25_top_k == 7
    assert config.graph_top_k == 9
    assert config.rrf_top_k == 12
    assert config.rerank_top_k == 4
    assert config.rrf_k == 50


@pytest.mark.parametrize(
    "field",
    [
        "dense_top_k",
        "bm25_top_k",
        "graph_top_k",
        "rrf_top_k",
        "rerank_top_k",
        "rrf_k",
    ],
)
def test_config_rejects_negative_values(field):
    values = {field: -1}

    with pytest.raises(ValueError):
        RetrievalConfig(**values)


def test_config_is_immutable():
    config = RetrievalConfig()

    with pytest.raises(Exception):
        config.dense_top_k = 100
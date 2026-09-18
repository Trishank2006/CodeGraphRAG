from datetime import datetime

import pytest

from evaluation.config import EvaluationConfig


def test_default_configuration():
    config = EvaluationConfig()

    assert config.dataset_version == "v1"
    assert config.top_k == 5
    assert config.rrf_k == 60
    assert config.rerank_top_k == 10
    assert config.embedding_model
    assert config.reranker_model
    assert config.llm_model
    assert config.max_context_items == 10
    assert config.max_context_characters == 12000
    assert config.repository_version == "unknown"
    assert config.benchmark_timestamp == ""


def test_create_configuration():
    config = EvaluationConfig.create(
        repository_version="abc123",
        top_k=10,
        rrf_k=50,
    )

    assert config.repository_version == "abc123"
    assert config.top_k == 10
    assert config.rrf_k == 50
    assert config.benchmark_timestamp

    timestamp = datetime.fromisoformat(
        config.benchmark_timestamp
    )

    assert timestamp.tzinfo is not None


def test_configuration_to_dict():
    config = EvaluationConfig(
        repository_version="abc123",
        benchmark_timestamp="2026-09-18T12:00:00+00:00",
    )

    data = config.to_dict()

    assert data["dataset_version"] == "v1"
    assert data["top_k"] == 5
    assert data["rrf_k"] == 60
    assert data["repository_version"] == "abc123"
    assert data["benchmark_timestamp"] == "2026-09-18T12:00:00+00:00"


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_k", 0),
        ("rrf_k", 0),
        ("rerank_top_k", 0),
        ("max_context_items", 0),
        ("max_context_characters", 0),
    ],
)
def test_invalid_positive_configuration(field, value):
    kwargs = {field: value}

    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)
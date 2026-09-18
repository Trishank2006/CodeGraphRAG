import pytest

from evaluation.generation_metrics import (
    answer_relevance,
    average_generation_metrics,
    context_relevance,
    faithfulness,
    generation_metrics,
)


def test_faithfulness_with_supported_answer():
    context = [
        "The application service is implemented in application/service.py.",
        "The service coordinates retrieval and answer generation.",
    ]

    answer = (
        "The application service is implemented in application/service.py "
        "and coordinates retrieval and answer generation."
    )

    score = faithfulness(answer, context)

    assert score > 0.5
    assert score <= 1.0


def test_faithfulness_with_empty_context():
    assert faithfulness("Some generated answer", []) == 0.0


def test_faithfulness_with_empty_answer():
    assert faithfulness("", ["Some context"]) == 0.0


def test_answer_relevance():
    query = "Where is the application service implemented?"
    answer = "The application service is implemented in application/service.py."

    score = answer_relevance(query, answer)

    assert score > 0.0
    assert score <= 1.0


def test_answer_relevance_unrelated_answer():
    query = "Where is the application service implemented?"
    answer = "The weather is sunny today."

    score = answer_relevance(query, answer)

    assert score < 0.5


def test_context_relevance():
    query = "Where is the application service implemented?"
    context = [
        "application/service.py contains the ApplicationService class.",
        "The service coordinates repository indexing and retrieval.",
    ]

    score = context_relevance(query, context)

    assert score > 0.0
    assert score <= 1.0


def test_context_relevance_empty_context():
    assert context_relevance("What is retrieval?", []) == 0.0


def test_generation_metrics():
    query = "Where is the retrieval pipeline?"
    answer = "The retrieval pipeline is implemented in retrieval/pipeline.py."
    context = [
        "retrieval/pipeline.py contains HybridRetrievalPipeline.",
    ]

    metrics = generation_metrics(query, answer, context)

    assert set(metrics) == {
        "faithfulness",
        "answer_relevance",
        "context_relevance",
    }

    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_average_generation_metrics():
    metrics = [
        {
            "faithfulness": 1.0,
            "answer_relevance": 0.8,
            "context_relevance": 0.6,
        },
        {
            "faithfulness": 0.5,
            "answer_relevance": 0.4,
            "context_relevance": 0.2,
        },
    ]

    averaged = average_generation_metrics(metrics)

    assert averaged["faithfulness"] == pytest.approx(0.75)
    assert averaged["answer_relevance"] == pytest.approx(0.60)
    assert averaged["context_relevance"] == pytest.approx(0.40)


def test_average_empty_metrics():
    assert average_generation_metrics([]) == {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_relevance": 0.0,
    }
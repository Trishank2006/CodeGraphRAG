import pytest

from evaluation.citation_metrics import (
    Citation,
    CitationEvaluation,
    calculate_citation_metrics,
    citation_coverage,
    citation_file_exists,
    citation_in_context,
    citation_line_range_valid,
    citation_supports_answer,
    citation_validity_rate,
    evaluate_citation,
)


def make_citation():
    return Citation(
        file_path="application/service.py",
        start_line=10,
        end_line=30,
    )


def make_context():
    return [
        (
            "application/service.py contains ApplicationService "
            "which coordinates retrieval and generation."
        ),
        "retrieval/pipeline.py contains HybridRetrievalPipeline.",
    ]


def test_citation_file_exists():
    citation = make_citation()

    assert citation_file_exists(
        citation,
        ["application/service.py", "api/main.py"],
    )

    assert not citation_file_exists(
        citation,
        ["api/main.py"],
    )


def test_citation_line_range_valid():
    assert citation_line_range_valid(
        Citation("file.py", 1, 10)
    )

    assert not citation_line_range_valid(
        Citation("file.py", 0, 10)
    )

    assert not citation_line_range_valid(
        Citation("file.py", 20, 10)
    )


def test_citation_in_context():
    citation = make_citation()

    assert citation_in_context(
        citation,
        make_context(),
    )

    assert not citation_in_context(
        Citation("missing.py", 1, 5),
        make_context(),
    )


def test_citation_supports_answer():
    citation = make_citation()

    answer = (
        "ApplicationService coordinates retrieval and generation."
    )

    assert citation_supports_answer(
        citation,
        answer,
        make_context(),
    )


def test_citation_supports_answer_without_context():
    citation = make_citation()

    assert not citation_supports_answer(
        citation,
        "ApplicationService coordinates retrieval.",
        [],
    )


def test_evaluate_citation():
    citation = make_citation()

    evaluation = evaluate_citation(
        citation=citation,
        available_files=["application/service.py"],
        context=make_context(),
        answer="ApplicationService coordinates retrieval and generation.",
    )

    assert evaluation.file_exists
    assert evaluation.line_range_valid
    assert evaluation.context_contains_source
    assert evaluation.source_supports_answer
    assert evaluation.valid


def test_invalid_citation():
    citation = Citation(
        file_path="missing.py",
        start_line=20,
        end_line=10,
    )

    evaluation = evaluate_citation(
        citation=citation,
        available_files=["application/service.py"],
        context=make_context(),
        answer="ApplicationService coordinates retrieval.",
    )

    assert not evaluation.file_exists
    assert not evaluation.line_range_valid
    assert not evaluation.context_contains_source
    assert not evaluation.valid


def test_citation_validity_rate():
    evaluations = [
        CitationEvaluation(True, True, True, True),
        CitationEvaluation(True, True, True, False),
        CitationEvaluation(True, True, True, True),
    ]

    assert citation_validity_rate(evaluations) == pytest.approx(2 / 3)


def test_empty_citation_validity_rate():
    assert citation_validity_rate([]) == 0.0


def test_citation_coverage():
    citations = [
        Citation(
            file_path="application/service.py",
            start_line=10,
            end_line=30,
        )
    ]

    answer = (
        "The application service is implemented in "
        "application/service.py."
    )

    score = citation_coverage(citations, answer)

    assert score > 0.0
    assert score <= 1.0


def test_empty_citation_coverage():
    assert citation_coverage([], "Some answer") == 0.0


def test_calculate_citation_metrics():
    evaluations = [
        CitationEvaluation(True, True, True, True),
        CitationEvaluation(True, True, True, True),
    ]

    citations = [
        Citation(
            file_path="application/service.py",
            start_line=10,
            end_line=30,
        )
    ]

    metrics = calculate_citation_metrics(
        evaluations,
        citations,
        "application/service.py contains ApplicationService.",
    )

    assert set(metrics) == {
        "citation_validity_rate",
        "citation_coverage",
    }

    assert metrics["citation_validity_rate"] == 1.0
    assert 0.0 <= metrics["citation_coverage"] <= 1.0
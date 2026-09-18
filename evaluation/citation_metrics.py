from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Citation:
    """Citation information produced for a generated answer."""

    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CitationEvaluation:
    """Detailed evaluation of one citation."""

    file_exists: bool
    line_range_valid: bool
    context_contains_source: bool
    source_supports_answer: bool

    @property
    def valid(self) -> bool:
        return (
            self.file_exists
            and self.line_range_valid
            and self.context_contains_source
            and self.source_supports_answer
        )


def citation_file_exists(
    citation: Citation,
    available_files: Iterable[str],
) -> bool:
    """Check whether the cited file exists in the available repository files."""
    return citation.file_path in set(available_files)


def citation_line_range_valid(citation: Citation) -> bool:
    """Check that a citation contains a valid source line range."""
    return (
        citation.start_line > 0
        and citation.end_line >= citation.start_line
    )


def citation_in_context(
    citation: Citation,
    context: Sequence[str] | Iterable[str],
) -> bool:
    """
    Check whether the cited file appears in retrieved context.

    Context items are expected to contain repository-relative file paths.
    """
    return any(
        citation.file_path in str(item)
        for item in context
    )


def citation_supports_answer(
    citation: Citation,
    answer: str,
    context: Sequence[str] | Iterable[str],
) -> bool:
    """
    Lightweight lexical support check.

    The cited file must occur in the context and at least one meaningful
    answer token must also occur in the cited context.
    """
    if not answer.strip():
        return False

    cited_context = [
        str(item)
        for item in context
        if citation.file_path in str(item)
    ]

    if not cited_context:
        return False

    answer_tokens = {
        token.lower()
        for token in answer.split()
        if len(token.strip(".,;:!?()[]{}\"'")) > 2
    }

    if not answer_tokens:
        return False

    context_text = " ".join(cited_context).lower()

    return any(
        token.strip(".,;:!?()[]{}\"'") in context_text
        for token in answer_tokens
    )


def evaluate_citation(
    citation: Citation,
    available_files: Iterable[str],
    context: Sequence[str] | Iterable[str],
    answer: str,
) -> CitationEvaluation:
    """Evaluate a single citation."""

    context_list = list(context)

    file_exists = citation_file_exists(
        citation,
        available_files,
    )

    line_range_valid = citation_line_range_valid(citation)

    context_contains_source = citation_in_context(
        citation,
        context_list,
    )

    source_supports_answer = citation_supports_answer(
        citation,
        answer,
        context_list,
    )

    return CitationEvaluation(
        file_exists=file_exists,
        line_range_valid=line_range_valid,
        context_contains_source=context_contains_source,
        source_supports_answer=source_supports_answer,
    )


def citation_validity_rate(
    evaluations: Sequence[CitationEvaluation],
) -> float:
    """Return the fraction of citations satisfying every validity condition."""
    if not evaluations:
        return 0.0

    valid_count = sum(
        evaluation.valid
        for evaluation in evaluations
    )

    return valid_count / len(evaluations)


def citation_coverage(
    citations: Sequence[Citation],
    answer: str,
) -> float:
    """
    Estimate how much of an answer is covered by cited files.

    A token is considered covered when it occurs in at least one cited
    file path. This metric is intentionally lightweight and deterministic.
    """
    if not answer.strip():
        return 0.0

    if not citations:
        return 0.0

    answer_tokens = {
        token.lower().strip(".,;:!?()[]{}\"'")
        for token in answer.split()
        if len(token.strip(".,;:!?()[]{}\"'")) > 2
    }

    if not answer_tokens:
        return 0.0

    cited_paths = " ".join(
        citation.file_path.lower()
        for citation in citations
    )

    covered = sum(
        token in cited_paths
        for token in answer_tokens
    )

    return covered / len(answer_tokens)


def calculate_citation_metrics(
    evaluations: Sequence[CitationEvaluation],
    citations: Sequence[Citation],
    answer: str,
) -> dict[str, float]:
    """Calculate citation validity and coverage metrics."""

    return {
        "citation_validity_rate": citation_validity_rate(evaluations),
        "citation_coverage": citation_coverage(citations, answer),
    }
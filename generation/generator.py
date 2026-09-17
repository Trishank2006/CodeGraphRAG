from __future__ import annotations

from typing import Protocol

from generation.context import ContextBuilder
from generation.llm import LLMClient, generate_text
from generation.models import Citation, GeneratedAnswer
from generation.prompt import PromptBuilder
from retrieval.models import RetrievalResult


class RetrievalPipeline(Protocol):
    """Interface required from the Phase 4 retrieval pipeline."""

    def search(self, query: str) -> list[RetrievalResult]:
        ...


class AnswerGenerator:
    """
    End-to-end Phase 5 answer generation pipeline.

    The generator coordinates:
        retrieval -> context assembly -> prompt construction -> LLM
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def answer_query(self, query: str) -> GeneratedAnswer:
        """
        Retrieve repository evidence and generate a grounded answer.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        # Phase 4 retrieval.
        retrieval_results = self.retrieval_pipeline.search(query)

        # Phase 5 context assembly.
        context = self.context_builder.build(
            query=query,
            results=retrieval_results,
        )

        # Build grounded prompt.
        prompt = self.prompt_builder.build_prompt(context)

        # Generate answer through provider-independent LLM interface.
        answer = generate_text(
            self.llm_client,
            prompt,
        )

        # Citations are generated from retrieved metadata rather than
        # asking the LLM to invent citation locations.
        citations = self._build_citations(context)

        return GeneratedAnswer(
            answer=answer,
            citations=tuple(citations),
        )

    @staticmethod
    def _build_citations(context) -> list[Citation]:
        """
        Convert selected context metadata into structured citations.

        Duplicate citations are removed while preserving context order.
        """
        citations: list[Citation] = []
        seen: set[tuple[str, str, int, int]] = set()

        for item in context.items:
            key = (
                item.file_path,
                item.symbol,
                item.start_line,
                item.end_line,
            )

            if key in seen:
                continue

            citations.append(
                Citation(
                    file_path=item.file_path,
                    symbol=item.symbol,
                    start_line=item.start_line,
                    end_line=item.end_line,
                )
            )

            seen.add(key)

        return citations


def answer_query(
    query: str,
    retrieval_pipeline: RetrievalPipeline,
    context_builder: ContextBuilder,
    prompt_builder: PromptBuilder,
    llm_client: LLMClient,
) -> GeneratedAnswer:
    """Convenience function for one-shot answer generation."""
    generator = AnswerGenerator(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    return generator.answer_query(query)
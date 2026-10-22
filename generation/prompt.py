
from __future__ import annotations

from generation.models import ContextPackage


class PromptBuilder:
    """Build grounded prompts from structured repository context."""

    SYSTEM_INSTRUCTIONS = """You are a codebase analysis assistant.

Answer the user's question using only the repository context provided.

Rules:
1. Do not invent files, functions, classes, variables, behavior, or relationships.
2. When making a claim about repository code, cite the relevant file and line range.
3. Support repository-specific claims with numbered evidence citations such as [1].
4. Distinguish direct code evidence from inferred relationships.
5. Use graph context only as structural evidence supplied by the repository.
6. If the provided context is insufficient, say: "Not found in the provided code."
7. Do not claim that code exists when it is not present in the supplied context.
8. Keep the answer concise and answer the specific question.
9. Do not invent line numbers or citation references.

The repository context is the source of truth.
"""

    def build_system_prompt(self) -> str:
        """Return the grounding and citation instructions."""
        return self.SYSTEM_INSTRUCTIONS

    def build_prompt(self, context: ContextPackage) -> str:
        """Build the complete prompt from repository context."""
        context_text = self._build_context_text(context)

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            "REPOSITORY CONTEXT AND EVIDENCE:\n"
            f"{context_text}\n\n"
            "USER QUESTION:\n"
            f"{context.query}\n\n"
            "Answer using the repository evidence above. Cite each "
            "repository-specific claim with the relevant file and line "
            "range, and use its numbered evidence reference, for example "
            "[1]. If the evidence is insufficient, respond: "
            '"Not found in the provided code."'
        )

    @staticmethod
    def _build_context_text(context: ContextPackage) -> str:
        """Format retrieved code and graph context deterministically."""
        sections: list[str] = []

        for index, item in enumerate(context.items, start=1):
            sections.append(
                f"[{index}] {item.file_path}\n"
                f"Symbol: {item.symbol}\n"
                f"Lines: {item.start_line}-{item.end_line}\n"
                f"Source: {item.source}\n"
                f"Code:\n{item.content}\n"
                f"[/ {index}]"
            )

        if not context.items:
            sections.append(
                "[NO CODE CONTEXT AVAILABLE]\n"
                "No repository code was retrieved for this query."
            )

        if context.graph_context:
            graph_lines = "\n".join(
                f"- {line}" for line in context.graph_context
            )
            sections.append(
                "[GRAPH CONTEXT]\n"
                f"{graph_lines}\n"
                "[/GRAPH CONTEXT]"
            )

        return "\n\n".join(sections)

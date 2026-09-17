from __future__ import annotations

from generation.models import ContextPackage


class PromptBuilder:
    """Build grounded prompts from structured repository context."""

    SYSTEM_INSTRUCTIONS = """You are a codebase analysis assistant.

Answer the user's question using only the repository context provided.

Rules:
1. Use the supplied repository evidence as the primary source of truth.
2. Do not invent files, functions, classes, variables, behavior, or relationships.
3. When making a claim about repository code, cite the relevant file and line range.
4. Distinguish direct code evidence from inferred relationships.
5. Use graph context only as structural evidence supplied by the repository.
6. If the provided context is insufficient to answer the question, explicitly say so.
7. Do not claim that code exists when it is not present in the supplied context.
"""

    def build_system_prompt(self) -> str:
        """Return the grounding and citation instructions."""
        return self.SYSTEM_INSTRUCTIONS

    def build_prompt(self, context: ContextPackage) -> str:
        """
        Build the complete user-facing prompt from a ContextPackage.
        """
        context_text = self._build_context_text(context)

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n"
            "REPOSITORY CONTEXT:\n"
            f"{context_text}\n\n"
            "USER QUESTION:\n"
            f"{context.query}\n\n"
            "Answer the question using the repository evidence above."
        )

    @staticmethod
    def _build_context_text(context: ContextPackage) -> str:
        """Format retrieved code and graph context deterministically."""
        sections: list[str] = []

        if context.items:
            for index, item in enumerate(context.items, start=1):
                sections.append(
                    f"[CONTEXT {index}]\n"
                    f"File: {item.file_path}\n"
                    f"Symbol: {item.symbol}\n"
                    f"Lines: {item.start_line}-{item.end_line}\n"
                    f"Source: {item.source}\n"
                    "Code:\n"
                    f"{item.content}\n"
                    f"[/CONTEXT {index}]"
                )
        else:
            sections.append(
                "[NO CODE CONTEXT AVAILABLE]\n"
                "No repository code was retrieved for this query."
            )

        if context.graph_context:
            graph_lines = "\n".join(
                f"- {line}"
                for line in context.graph_context
            )

            sections.append(
                "[GRAPH CONTEXT]\n"
                f"{graph_lines}\n"
                "[/GRAPH CONTEXT]"
            )

        return "\n\n".join(sections)
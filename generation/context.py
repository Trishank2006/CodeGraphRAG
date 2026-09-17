from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from generation.models import ContextItem, ContextPackage
from retrieval.models import RetrievalResult


GraphContextGetter = Callable[[str, int], dict[str, Any]]


@dataclass(frozen=True)
class ContextConfig:
    """Limits for the amount of evidence sent to the LLM."""

    max_items: int = 10
    max_characters: int = 30000
    graph_hops: int = 1

    def __post_init__(self) -> None:
        if self.max_items < 0:
            raise ValueError("max_items must be non-negative")

        if self.max_characters < 0:
            raise ValueError("max_characters must be non-negative")

        if self.graph_hops < 1:
            raise ValueError("graph_hops must be at least 1")


class ContextBuilder:
    """
    Convert Phase 4 retrieval results into bounded generation context.

    The existing Phase 4 ranking order is preserved. ContextBuilder does
    not introduce a new relevance-ranking mechanism.
    """

    def __init__(
        self,
        config: ContextConfig | None = None,
        graph_context_getter: GraphContextGetter | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        self.graph_context_getter = graph_context_getter

    def build(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> ContextPackage:
        """
        Build a bounded ContextPackage from ranked retrieval results.
        """
        if not query.strip() or not results:
            return ContextPackage(query=query)

        selected_items: list[ContextItem] = []
        seen_ids: set[str] = set()
        seen_content: set[tuple[str, str]] = set()

        characters_used = 0

        for result in results:
            if len(selected_items) >= self.config.max_items:
                break

            if not result.content.strip():
                continue

            # Prevent the same retrieval result from appearing twice.
            if result.id in seen_ids:
                continue

            # Prevent identical code from appearing multiple times even
            # when the same content comes from different retrieval IDs
            # or retrieval sources.
            content_key = (
                result.symbol,
                result.content,
            )

            if content_key in seen_content:
                continue

            item = ContextItem(
                id=result.id,
                file_path=result.file_path,
                symbol=result.symbol,
                start_line=result.start_line,
                end_line=result.end_line,
                content=result.content,
                source=result.source,
            )

            item_characters = self._item_character_cost(item)

            if (
                characters_used + item_characters
                > self.config.max_characters
            ):
                continue

            selected_items.append(item)
            seen_ids.add(result.id)
            seen_content.add(content_key)
            characters_used += item_characters

        graph_context = self._build_graph_context(
            selected_items
        )

        return ContextPackage(
            query=query,
            items=tuple(selected_items),
            graph_context=tuple(graph_context),
        )

    @staticmethod
    def _item_character_cost(item: ContextItem) -> int:
        """
        Estimate the context budget consumed by one item.

        Metadata is counted because it will also be included in the
        generated prompt later.
        """
        metadata = (
            f"File: {item.file_path}\n"
            f"Symbol: {item.symbol}\n"
            f"Lines: {item.start_line}-{item.end_line}\n"
            f"Source: {item.source}\n"
        )

        return len(metadata) + len(item.content)

    def _build_graph_context(
        self,
        items: list[ContextItem],
    ) -> list[str]:
        """
        Convert graph-context API results into deterministic text lines.
        """
        if self.graph_context_getter is None:
            return []

        graph_lines: list[str] = []
        seen_lines: set[str] = set()

        for item in items:
            graph_data = self.graph_context_getter(
                item.id,
                self.config.graph_hops,
            )

            connections = graph_data.get("connections", [])

            for connection in connections:
                target_name = connection.get("target_name")
                relationship = connection.get("relationship")
                target_file = connection.get("target_file")
                distance = connection.get("distance")

                if not target_name or not relationship:
                    continue

                line = (
                    f"{item.symbol} "
                    f"--{relationship}--> "
                    f"{target_name}"
                )

                if target_file:
                    line += f" ({target_file})"

                if distance is not None:
                    line += f" [distance={distance}]"

                if line not in seen_lines:
                    graph_lines.append(line)
                    seen_lines.add(line)

        return graph_lines
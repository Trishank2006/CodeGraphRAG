
from pathlib import Path
from typing import List, Optional

from retrieval.models import RetrievalResult
from graph.queries import GraphQueries
from graph.neo4j_store import Neo4jStore
from graph.ranking import calculate_graph_score


class GraphRetriever:
    """Graph retrieval engine with source-code enrichment."""

    def __init__(
        self,
        store: Neo4jStore,
        repository_root: Optional[str] = None,
    ):
        self.store = store
        self.queries = GraphQueries(store)
        self.repository_root = Path(
            repository_root or "."
        ).resolve()

    def _safe_int(self, value, default: int = 0) -> int:
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def _resolve_source_path(self, file_path: str) -> Optional[Path]:
        """Resolve a graph file path within the configured repository."""
        if not file_path:
            return None

        path = Path(file_path)

        candidates = (
            [path]
            if path.is_absolute()
            else [
                self.repository_root / path,
                Path.cwd() / path,
            ]
        )

        for candidate in candidates:
            try:
                resolved = candidate.resolve()

                # Avoid reading arbitrary files outside the configured
                # repository root.
                resolved.relative_to(self.repository_root)

                if resolved.is_file():
                    return resolved
            except (OSError, RuntimeError, ValueError):
                continue

        return None

    def _read_source_content(
        self,
        file_path: str,
        start_line: int = 0,
        end_line: int = 0,
        max_lines: int = 120,
    ) -> str:
        """Read a bounded source snippet using 1-based line numbers."""
        source_path = self._resolve_source_path(file_path)

        if source_path is None:
            return ""

        try:
            with source_path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as source_file:
                lines = source_file.readlines()
        except OSError:
            return ""

        if not lines:
            return ""

        if start_line > 0:
            start = min(start_line - 1, len(lines))
        else:
            start = 0

        if end_line >= start_line and end_line > 0:
            end = min(end_line, len(lines))
        else:
            end = min(start + max_lines, len(lines))

        # Bound content length to avoid oversized reranker inputs.
        end = min(end, start + max_lines)
        content = "".join(lines[start:end]).strip()

        # Fall back to a short file excerpt if graph line metadata
        # is missing or points outside the file.
        if not content and (start_line <= 0 or end_line <= 0):
            content = "".join(lines[:max_lines]).strip()

        return content

    def _node_to_retrieval_result(
        self,
        node_data: dict,
        score: float = 1.0,
        content: Optional[str] = None,
    ) -> RetrievalResult:
        """Convert a graph node into a retrieval result with code content."""
        file_path = str(node_data.get("file_path") or "")
        start_line = self._safe_int(node_data.get("start_line"))
        end_line = self._safe_int(node_data.get("end_line"))

        if content is None:
            content = self._read_source_content(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
            )

        return RetrievalResult(
            id=str(node_data.get("id") or ""),
            source="graph",
            score=float(score),
            repository=str(
                node_data.get("repository") or "default_repo"
            ),
            file_path=file_path,
            language=str(node_data.get("language") or "unknown"),
            symbol=str(node_data.get("name") or ""),
            start_line=start_line,
            end_line=end_line,
            content=content,
        )

    def get_callers(
        self,
        entity_id_or_name: str,
        repository: Optional[str] = None,
    ) -> List[RetrievalResult]:
        nodes = self.queries.get_function_callers(
            entity_id_or_name,
            repository=repository,
        )
        score = calculate_graph_score("CALLS", distance=1)
        return [
            self._node_to_retrieval_result(node, score=score)
            for node in nodes
        ]

    def get_callees(
        self,
        entity_id_or_name: str,
        repository: Optional[str] = None,
    ) -> List[RetrievalResult]:
        nodes = self.queries.get_function_callees(
            entity_id_or_name,
            repository=repository,
        )
        score = calculate_graph_score("CALLS", distance=1)
        return [
            self._node_to_retrieval_result(node, score=score)
            for node in nodes
        ]

    def get_dependencies(
        self,
        file_path_or_symbol: str,
        repository: Optional[str] = None,
    ) -> List[RetrievalResult]:
        nodes = self.queries.get_dependencies(
            file_path_or_symbol,
            repository=repository,
        )
        score = calculate_graph_score("IMPORTS", distance=1)
        return [
            self._node_to_retrieval_result(node, score=score)
            for node in nodes
        ]

    def get_parent_classes(
        self,
        class_name_or_id: str,
        repository: Optional[str] = None,
    ) -> List[RetrievalResult]:
        nodes = self.queries.get_parent_classes(
            class_name_or_id,
            repository=repository,
        )
        score = calculate_graph_score("INHERITS", distance=1)
        return [
            self._node_to_retrieval_result(node, score=score)
            for node in nodes
        ]

    def get_file_symbols(
        self,
        file_path: str,
        repository: Optional[str] = None,
    ) -> List[RetrievalResult]:
        nodes = self.queries.get_file_symbols(
            file_path,
            repository=repository,
        )
        score = calculate_graph_score("CONTAINS", distance=1)
        return [
            self._node_to_retrieval_result(node, score=score)
            for node in nodes
        ]

    def get_related_code(
        self,
        entity_id_or_name: str,
        hops: int = 1,
        relationship_types: Optional[List[str]] = None,
        repository: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieve related entities with relationship-aware scores."""
        hops = min(max(hops, 1), 2)

        nodes = self.queries.get_neighbors_with_depth(
            entity_id_or_name,
            max_hops=hops,
            relationship_types=relationship_types,
            repository=repository,
        )

        results: List[RetrievalResult] = []

        for node in nodes:
            distance = node.get("distance", 1)
            relationship = node.get("relationship", "CALLS")
            score = calculate_graph_score(
                relationship,
                distance=distance,
            )

            results.append(
                self._node_to_retrieval_result(
                    node,
                    score=score,
                )
            )

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        hops: int = 1,
        entity_type: Optional[str] = None,
        language: Optional[str] = None,
        repository: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Find seeds, expand neighbors, enrich candidates, and rank."""
        if top_k <= 0 or not query or not query.strip():
            return []

        seeds = self.queries.find_entities(
            query,
            entity_type=entity_type,
            language=language,
            repository=repository,
        )

        if not seeds:
            return []

        results: List[RetrievalResult] = []
        seen_ids: set[str] = set()

        # Include unique seed entities.
        for seed in seeds:
            seed_id = str(seed.get("id") or "")

            if not seed_id or seed_id in seen_ids:
                continue

            results.append(
                self._node_to_retrieval_result(
                    seed,
                    score=1.0,
                )
            )
            seen_ids.add(seed_id)

        # Collect neighbors from all seeds before truncating.
        for seed in seeds:
            seed_name = seed.get("name")

            if not seed_name:
                continue

            neighbors = self.get_related_code(
                str(seed_name),
                hops=hops,
                repository=repository,
            )

            for neighbor in neighbors:
                if not neighbor.id or neighbor.id in seen_ids:
                    continue

                results.append(neighbor)
                seen_ids.add(neighbor.id)

        # Deterministic ordering.
        results.sort(
            key=lambda result: (-result.score, result.id)
        )

        return results[:top_k]


def graph_search(
    query: str,
    store: Optional[Neo4jStore] = None,
    top_k: int = 10,
    hops: int = 1,
    entity_type: Optional[str] = None,
    language: Optional[str] = None,
    repository: Optional[str] = None,
    repository_root: Optional[str] = None,
) -> List[RetrievalResult]:
    """Convenience API for graph-based code search."""
    owns_store = store is None
    target_store = store if store is not None else Neo4jStore()

    try:
        retriever = GraphRetriever(
            target_store,
            repository_root=repository_root,
        )

        return retriever.search(
            query=query,
            top_k=top_k,
            hops=hops,
            entity_type=entity_type,
            language=language,
            repository=repository,
        )
    finally:
        if owns_store:
            target_store.close()

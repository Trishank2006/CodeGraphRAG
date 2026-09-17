from typing import List, Optional
from retrieval.models import RetrievalResult
from graph.queries import GraphQueries
from graph.neo4j_store import Neo4jStore
from graph.ranking import calculate_graph_score


class GraphRetriever:
    """Graph Retrieval Engine that maps Neo4j structural nodes to standardized RetrievalResult."""

    def __init__(self, store: Neo4jStore):
        self.store = store
        self.queries = GraphQueries(store)

    def _node_to_retrieval_result(
        self, node_data: dict, score: float = 1.0, content: str = ""
    ) -> RetrievalResult:
        def _safe_int(val, default: int = 0) -> int:
            try:
                return int(val) if val is not None else default
            except (ValueError, TypeError):
                return default

        return RetrievalResult(
            id=str(node_data.get("id") or ""),
            source="graph",
            score=float(score),
            repository=str(node_data.get("repository") or "default_repo"),
            file_path=str(node_data.get("file_path") or ""),
            language=str(node_data.get("language") or "unknown"),
            symbol=str(node_data.get("name") or ""),
            start_line=_safe_int(node_data.get("start_line"), 0),
            end_line=_safe_int(node_data.get("end_line"), 0),
            content=content,
        )

    def get_callers(self, entity_id_or_name: str) -> List[RetrievalResult]:
        raw_nodes = self.queries.get_function_callers(entity_id_or_name)
        score = calculate_graph_score("CALLS", distance=1)
        return [self._node_to_retrieval_result(n, score=score) for n in raw_nodes]

    def get_callees(self, entity_id_or_name: str) -> List[RetrievalResult]:
        raw_nodes = self.queries.get_function_callees(entity_id_or_name)
        score = calculate_graph_score("CALLS", distance=1)
        return [self._node_to_retrieval_result(n, score=score) for n in raw_nodes]

    def get_dependencies(self, file_path_or_symbol: str) -> List[RetrievalResult]:
        raw_nodes = self.queries.get_dependencies(file_path_or_symbol)
        score = calculate_graph_score("IMPORTS", distance=1)
        return [self._node_to_retrieval_result(n, score=score) for n in raw_nodes]

    def get_parent_classes(self, class_name_or_id: str) -> List[RetrievalResult]:
        raw_nodes = self.queries.get_parent_classes(class_name_or_id)
        score = calculate_graph_score("INHERITS", distance=1)
        return [self._node_to_retrieval_result(n, score=score) for n in raw_nodes]

    def get_file_symbols(self, file_path: str) -> List[RetrievalResult]:
        raw_nodes = self.queries.get_file_symbols(file_path)
        score = calculate_graph_score("CONTAINS", distance=1)
        return [self._node_to_retrieval_result(n, score=score) for n in raw_nodes]

    def get_related_code(
        self,
        entity_id_or_name: str,
        hops: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Bounded graph traversal retrieving related entities with relationship-aware scoring."""
        hops = min(max(hops, 1), 2)
        raw_nodes = self.queries.get_neighbors_with_depth(
            entity_id_or_name, max_hops=hops, relationship_types=relationship_types
        )

        results = []
        for item in raw_nodes:
            distance = item.get("distance", 1)
            rel_type = item.get("relationship", "CALLS")
            score = calculate_graph_score(rel_type, distance=distance)
            results.append(self._node_to_retrieval_result(item, score=score))
        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        hops: int = 1,
        entity_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Main graph search interface: finds seed entities then expands neighborhood with ranking."""
        seed_entities = self.queries.find_entities(
            query, entity_type=entity_type, language=language
        )
        if not seed_entities:
            return []

        results: List[RetrievalResult] = []
        seen_ids = set()

        for seed in seed_entities:
            if seed["id"] not in seen_ids:
                results.append(self._node_to_retrieval_result(seed, score=1.0))
                seen_ids.add(seed["id"])

        for seed in seed_entities:
            neighbors = self.get_related_code(seed["name"], hops=hops)
            for neighbor in neighbors:
                if neighbor.id not in seen_ids:
                    results.append(neighbor)
                    seen_ids.add(neighbor.id)
                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break

        # Deterministic ordering based on structural relevance score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def graph_search(
    query: str,
    store: Optional[Neo4jStore] = None,
    top_k: int = 10,
    hops: int = 1,
    entity_type: Optional[str] = None,
    language: Optional[str] = None,
) -> List[RetrievalResult]:
    """Convenience functional API mirroring Person 1's search_code()."""
    target_store = store or Neo4jStore()
    retriever = GraphRetriever(target_store)
    return retriever.search(
        query=query,
        top_k=top_k,
        hops=hops,
        entity_type=entity_type,
        language=language,
    )
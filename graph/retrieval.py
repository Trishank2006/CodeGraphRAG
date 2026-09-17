from typing import List, Optional
from retrieval.models import RetrievalResult
from graph.queries import GraphQueries
from graph.neo4j_store import Neo4jStore


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
        return [self._node_to_retrieval_result(n, score=1.0) for n in raw_nodes]

    def get_callees(self, entity_id_or_name: str) -> List[RetrievalResult]:
        raw_nodes = self.queries.get_function_callees(entity_id_or_name)
        return [self._node_to_retrieval_result(n, score=1.0) for n in raw_nodes]

    def get_dependencies(self, file_path_or_symbol: str) -> List[RetrievalResult]:
        raw_nodes = self.queries.get_dependencies(file_path_or_symbol)
        return [self._node_to_retrieval_result(n, score=0.8) for n in raw_nodes]

    def get_related_code(
        self, entity_id_or_name: str, hops: int = 1
    ) -> List[RetrievalResult]:
        """Bounded graph traversal retrieving related entities based on hop distance."""
        hops = min(max(hops, 1), 2)  # Strict bound: 1 or 2 hops
        raw_nodes = self.queries.get_neighbors_with_depth(entity_id_or_name, max_hops=hops)

        results = []
        for item in raw_nodes:
            distance = item.get("distance", 1)
            score = 1.0 if distance == 1 else 0.7
            results.append(self._node_to_retrieval_result(item, score=score))
        return results

    def search(self, query: str, top_k: int = 10, hops: int = 1) -> List[RetrievalResult]:
        """Main graph search interface: finds seed entities then gathers related code."""
        seed_entities = self.queries.find_entities(query)
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

        return results[:top_k]


def graph_search(
    query: str,
    store: Optional[Neo4jStore] = None,
    top_k: int = 10,
    hops: int = 1,
) -> List[RetrievalResult]:
    """Convenience functional API mirroring Person 1's search_code()."""
    target_store = store or Neo4jStore()
    retriever = GraphRetriever(target_store)
    return retriever.search(query=query, top_k=top_k, hops=hops)
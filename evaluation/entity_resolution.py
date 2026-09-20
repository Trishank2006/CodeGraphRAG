from typing import List, Dict, Any, Optional
from graph.queries import GraphQueries


class EntityResolutionEvaluator:
    """Evaluates the graph's capability to resolve ambiguous symbol queries across repositories."""

    def __init__(self, queries: GraphQueries):
        self.queries = queries

    def evaluate_symbol_lookup(
        self,
        query: str,
        expected_id: str,
        repository: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tests exact and fuzzy symbol resolution with optional repository isolation."""
        records = self.queries.find_entities(query=query, repository=repository)
        if not records:
            return {
                "query": query,
                "found": False,
                "resolved_id": None,
                "is_correct": False,
                "is_isolated": True,
            }

        top_match = records[0]
        resolved_id = top_match.get("id")
        resolved_repo = top_match.get("repository")

        is_isolated = True
        if repository is not None:
            is_isolated = resolved_repo == repository

        return {
            "query": query,
            "found": True,
            "resolved_id": resolved_id,
            "is_correct": resolved_id == expected_id,
            "is_isolated": is_isolated,
        }
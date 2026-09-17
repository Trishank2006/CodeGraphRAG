from typing import List, Optional, Dict, Any
from graph.neo4j_store import Neo4jStore
from graph.context import GraphContextExtractor
from graph.path import GraphPathFinder
from graph.queries import GraphQueries
from graph.evidence import GraphEvidence, GraphPath, EntityMetadata


class GraphGroundingService:
    """Unified service bridging Neo4j structural knowledge to the generation layer."""

    def __init__(self, store: Optional[Neo4jStore] = None):
        self.store = store or Neo4jStore()
        self.extractor = GraphContextExtractor(self.store)
        self.path_finder = GraphPathFinder(self.store)
        self.queries = GraphQueries(self.store)

    def get_evidence(self, entity_id_or_name: str, hops: int = 1) -> List[GraphEvidence]:
        """Extracts structured GraphEvidence instances with exact file and line locations."""
        ctx = self.extractor.get_graph_context(entity_id_or_name, hops=hops)
        source_name = ctx.get("name") or entity_id_or_name
        source_file = ctx.get("file_path", "")
        source_start = ctx.get("start_line", 0)
        source_end = ctx.get("end_line", 0)

        evidence_items: List[GraphEvidence] = []
        for conn in ctx.get("connections", []):
            evidence_items.append(
                GraphEvidence(
                    source_entity=source_name,
                    relationship=conn.get("relationship", "CONNECTED_TO"),
                    target_entity=conn.get("target_name") or conn.get("target_id", "Unknown"),
                    distance=conn.get("distance", 1),
                    source_file=source_file,
                    target_file=conn.get("target_file", ""),
                    source_start_line=source_start,
                    source_end_line=source_end,
                    target_start_line=conn.get("target_start", 0) or 0,
                    target_end_line=conn.get("target_end", 0) or 0,
                )
            )
        return evidence_items

    def get_context(self, entity_id_or_name: str, hops: int = 1) -> Dict[str, Any]:
        """Returns structured dictionary context for ContextBuilder."""
        return self.extractor.get_graph_context(entity_id_or_name, hops=hops)

    def format_context_tree(self, entity_id_or_name: str, hops: int = 1) -> str:
        """Returns human/LLM-readable text tree representation."""
        return self.extractor.format_hierarchy(entity_id_or_name, hops=hops)

    def get_path(self, source: str, target: str, max_depth: int = 3) -> Optional[GraphPath]:
        """Returns end-to-end execution path between two entities."""
        return self.path_finder.get_path(source, target, max_depth=max_depth)

    def get_mode_aware_context(self, entity: str, query: str) -> str:
        """Inspects query intent to customize structural context presentation."""
        query_lower = query.lower()
        if any(w in query_lower for w in ["who calls", "callers", "invoked by"]):
            callers = self.queries.get_function_callers(entity)
            if not callers:
                return f"No callers discovered for {entity}."
            lines = [f"Callers of {entity}:"]
            for c in callers:
                lines.append(f"  ← {c.get('name')} ({c.get('file_path')})")
            return "\n".join(lines)

        if any(w in query_lower for w in ["inherits", "subclass", "base class"]):
            parents = self.queries.get_parent_classes(entity)
            if not parents:
                return f"No inheritance hierarchy discovered for {entity}."
            lines = [f"Inheritance for {entity}:"]
            for p in parents:
                lines.append(f"  → INHERITS {p.get('name')} ({p.get('file_path')})")
            return "\n".join(lines)

        # Default: structural hierarchy tree
        return self.format_context_tree(entity, hops=1)
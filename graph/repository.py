from typing import Dict, Any
from graph.neo4j_store import Neo4jStore


class GraphRepositoryManager:
    """Manages repository isolation, statistics, and deletion in the graph."""

    def __init__(self, store: Neo4jStore | None = None):
        self.store = store or Neo4jStore()

    def delete_repository(self, repository: str) -> int:
        """Deletes all nodes and relationships associated with a repository."""
        cypher = """
        MATCH (n:Node {repository: $repo})
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """
        records = self.store.execute_query(cypher, {"repo": repository})
        return records[0].get("deleted_count", 0) if records else 0

    def get_repository_stats(self, repository: str) -> Dict[str, Any]:
        """Returns entity and relationship metrics for a specific repository."""
        cypher = """
        MATCH (n:Node {repository: $repo})
        OPTIONAL MATCH (n)-[r]->(m:Node {repository: $repo})
        RETURN count(DISTINCT n) AS node_count,
               count(DISTINCT r) AS rel_count,
               collect(DISTINCT n.label) AS labels
        """
        records = self.store.execute_query(cypher, {"repo": repository})
        if not records:
            return {"repository": repository, "node_count": 0, "rel_count": 0, "labels": []}

        rec = records[0]
        return {
            "repository": repository,
            "node_count": rec.get("node_count", 0),
            "rel_count": rec.get("rel_count", 0),
            "labels": [l for l in rec.get("labels", []) if l is not None],
        }
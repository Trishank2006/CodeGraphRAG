from typing import List, Dict, Any, Optional
from graph.neo4j_store import Neo4jStore


class GraphPathFinder:
    """Traces execution and dependency paths between code entities."""

    def __init__(self, store: Neo4jStore):
        self.store = store

    def find_relationship(
        self, source_name_or_id: str, target_name_or_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        max_depth = min(max(max_depth, 1), 5)
        cypher = f"""
        MATCH (start:Node), (target:Node)
        WHERE (start.id = $src OR start.name = $src)
          AND (target.id = $tgt OR target.name = $tgt)
        MATCH p = shortestPath((start)-[r:RELATION*..{max_depth}]-(target))
        RETURN [n in nodes(p) | {{id: n.id, name: n.name, label: n.label, file_path: n.file_path}}] AS nodes,
               [rel in relationships(p) | rel.type] AS relationships,
               length(p) AS path_length
        """
        records = self.store.execute_query(
            cypher, {"src": source_name_or_id, "tgt": target_name_or_id}
        )
        if not records:
            return []

        rec = records[0]
        nodes = rec.get("nodes", [])
        rels = rec.get("relationships", [])

        steps = []
        for i in range(len(rels)):
            steps.append({
                "from": nodes[i],
                "relationship": rels[i],
                "to": nodes[i + 1] if i + 1 < len(nodes) else None,
            })
        return steps
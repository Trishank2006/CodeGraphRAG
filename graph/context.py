from typing import Dict, Any, List
from graph.neo4j_store import Neo4jStore


class GraphContextExtractor:
    """Extracts hierarchical structural context around code entities for generation."""

    def __init__(self, store: Neo4jStore):
        self.store = store

    def get_graph_context(self, entity_id_or_name: str, hops: int = 1) -> Dict[str, Any]:
        hops = min(max(hops, 1), 2)
        cypher = f"""
        MATCH (root:Node)
        WHERE root.id = $val OR root.name = $val
        OPTIONAL MATCH path = (root)-[r:RELATION*1..{hops}]-(target:Node)
        RETURN root.id AS root_id, root.name AS root_name, root.label AS root_type,
               root.file_path AS root_file,
               collect(DISTINCT {{
                   target_id: target.id,
                   target_name: target.name,
                   target_type: target.label,
                   target_file: target.file_path,
                   relationship: last(relationships(path)).type,
                   distance: length(path)
               }}) AS connections
        LIMIT 1
        """
        records = self.store.execute_query(cypher, {"val": entity_id_or_name})
        if not records:
            return {"entity": entity_id_or_name, "connections": []}

        rec = records[0]
        # Clean null targets from empty OPTIONAL MATCH
        conns = [c for c in rec.get("connections", []) if c.get("target_id") is not None]
        return {
            "entity_id": rec.get("root_id"),
            "name": rec.get("root_name"),
            "type": rec.get("root_type"),
            "file_path": rec.get("root_file"),
            "connections": conns,
        }
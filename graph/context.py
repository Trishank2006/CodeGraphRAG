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
               root.file_path AS root_file, root.language AS root_lang,
               root.start_line AS root_start, root.end_line AS root_end,
               coalesce(root.repository, 'default_repo') AS root_repo,
               collect(DISTINCT {{
                   target_id: target.id,
                   target_name: target.name,
                   target_type: target.label,
                   target_file: target.file_path,
                   target_start: target.start_line,
                   target_end: target.end_line,
                   relationship: last(relationships(path)).type,
                   distance: length(path)
               }}) AS connections
        LIMIT 1
        """
        records = self.store.execute_query(cypher, {"val": entity_id_or_name})
        if not records:
            return {
                "entity": entity_id_or_name,
                "name": entity_id_or_name,
                "connections": [],
            }

        rec = records[0]
        raw_conns = rec.get("connections", [])
        # Filter out null targets from empty OPTIONAL MATCH
        conns = [c for c in raw_conns if c.get("target_id") is not None]

        return {
            "entity_id": rec.get("root_id"),
            "name": rec.get("root_name"),
            "type": rec.get("root_type"),
            "file_path": rec.get("root_file", ""),
            "start_line": rec.get("root_start", 0) or 0,
            "end_line": rec.get("root_end", 0) or 0,
            "connections": conns,
        }

    def format_hierarchy(self, entity_id_or_name: str, hops: int = 1) -> str:
        """Renders structural hierarchy as clean ASCII trees for LLM prompt injection."""
        ctx = self.get_graph_context(entity_id_or_name, hops=hops)
        root_name = ctx.get("name") or entity_id_or_name
        lines = [f"{root_name}"]

        connections = ctx.get("connections", [])
        if not connections:
            lines.append("  (no direct structural connections discovered)")
            return "\n".join(lines)

        for i, conn in enumerate(connections):
            is_last = i == len(connections) - 1
            prefix = "└── " if is_last else "├── "
            rel = conn.get("relationship", "CONNECTED_TO")
            tgt = conn.get("target_name") or conn.get("target_id", "Unknown")
            lines.append(f"{prefix}{rel} → {tgt}")

        return "\n".join(lines)
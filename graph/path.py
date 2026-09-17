from typing import List, Dict, Any, Optional
from graph.neo4j_store import Neo4jStore
from graph.evidence import EntityMetadata, PathStep, GraphPath


class GraphPathFinder:
    """Traces execution and dependency paths between code entities for grounded generation."""

    def __init__(self, store: Neo4jStore):
        self.store = store

    def find_relationship(
        self, source_name_or_id: str, target_name_or_id: str, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Returns raw path steps for backward compatibility with Phase 4 callers and unit tests."""
        max_depth = min(max(max_depth, 1), 5)
        cypher = f"""
        MATCH (start:Node), (target:Node)
        WHERE (start.id = $src OR start.name = $src)
          AND (target.id = $tgt OR target.name = $tgt)
        MATCH p = shortestPath((start)-[r:RELATION*..{max_depth}]-(target))
        RETURN [n in nodes(p) | {{
            id: n.id,
            name: n.name,
            label: n.label,
            file_path: n.file_path,
            start_line: coalesce(n.start_line, 0),
            end_line: coalesce(n.end_line, 0)
        }}] AS nodes,
        [rel in relationships(p) | rel.type] AS relationships,
        length(p) AS path_length
        LIMIT 1
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

    def get_path(
        self, source_name_or_id: str, target_name_or_id: str, max_depth: int = 3
    ) -> Optional[GraphPath]:
        """Returns a structured GraphPath model with rich source metadata for generation."""
        raw_steps = self.find_relationship(
            source_name_or_id, target_name_or_id, max_depth=max_depth
        )
        if not raw_steps:
            return None

        path_steps: List[PathStep] = []
        for s in raw_steps:
            from_node = s["from"]
            to_node = s["to"]
            if not from_node or not to_node:
                continue

            src_meta = EntityMetadata(
                id=str(from_node.get("id") or ""),
                name=str(from_node.get("name") or ""),
                label=str(from_node.get("label") or "Node"),
                file_path=str(from_node.get("file_path") or ""),
                start_line=int(from_node.get("start_line") or 0),
                end_line=int(from_node.get("end_line") or 0),
            )
            tgt_meta = EntityMetadata(
                id=str(to_node.get("id") or ""),
                name=str(to_node.get("name") or ""),
                label=str(to_node.get("label") or "Node"),
                file_path=str(to_node.get("file_path") or ""),
                start_line=int(to_node.get("start_line") or 0),
                end_line=int(to_node.get("end_line") or 0),
            )
            path_steps.append(
                PathStep(source=src_meta, relationship=s["relationship"], target=tgt_meta)
            )

        return GraphPath(
            source=source_name_or_id,
            target=target_name_or_id,
            steps=path_steps,
            path_length=len(path_steps),
        )
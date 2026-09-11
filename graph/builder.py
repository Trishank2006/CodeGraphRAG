from typing import List, Tuple
from parser.models import ParsedFile
from graph.models import GraphNode, GraphEdge
from graph.schema import NodeType, RelationType


class GraphBuilder:
    """Converts AST parsed files into structured graph nodes and edges."""

    def __init__(self, repository_name: str = "default_repo"):
        self.repository_name = repository_name

    def build_from_parsed_files(
        self, parsed_files: List[ParsedFile]
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        seen_node_ids = set()

        for pf in parsed_files:
            # 1. Create File node
            file_node_id = f"file:{pf.file_path}"
            if file_node_id not in seen_node_ids:
                nodes.append(
                    GraphNode(
                        id=file_node_id,
                        label=NodeType.FILE,
                        name=pf.file_path.split("/")[-1],
                        file_path=pf.file_path,
                        language=pf.language,
                        properties={"repository": self.repository_name},
                    )
                )
                seen_node_ids.add(file_node_id)

            # 2. Create Entity nodes (Class, Function)
            for entity in pf.entities:
                node_label = (
                    NodeType.CLASS if entity.type == "class" else NodeType.FUNCTION
                )
                if entity.id not in seen_node_ids:
                    nodes.append(
                        GraphNode(
                            id=entity.id,
                            label=node_label,
                            name=entity.name,
                            file_path=entity.file_path,
                            language=pf.language,
                            start_line=entity.start_line,
                            end_line=entity.end_line,
                            properties={"repository": self.repository_name},
                        )
                    )
                    seen_node_ids.add(entity.id)

            # 3. Create Edges
            for rel in pf.relationships:
                # Convert file_path reference to standard file node id if applicable
                source = (
                    f"file:{rel.source_id}"
                    if rel.source_id == pf.file_path
                    else rel.source_id
                )
                edges.append(
                    GraphEdge(
                        source_id=source,
                        target_id=rel.target_id,
                        type=RelationType(rel.type),
                    )
                )

        return nodes, edges
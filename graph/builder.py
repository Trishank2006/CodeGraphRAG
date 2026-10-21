
from typing import List, Tuple

from parser.models import ParsedFile
from graph.models import GraphNode, GraphEdge
from graph.schema import NodeType, RelationType


class GraphBuilder:
    """Build a repository-scoped graph from AST-parsed files."""

    def __init__(self, repository_name: str = "default_repo"):
        if not repository_name or not repository_name.strip():
            raise ValueError("Repository name must not be empty")

        self.repository_name = repository_name

    def _scoped_id(self, node_id: str) -> str:
        """Make node IDs unique across repositories."""
        return f"repo:{self.repository_name}:{node_id}"

    def build_from_parsed_files(
        self, parsed_files: List[ParsedFile]
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        seen_node_ids = set()

        def add_reference_node(
            reference_id: str,
            file_path: str,
            language: str,
        ) -> None:
            scoped_id = self._scoped_id(reference_id)

            if scoped_id in seen_node_ids:
                return

            if reference_id.startswith("module:"):
                label = NodeType.MODULE
                name = reference_id.removeprefix("module:")
            else:
                label = NodeType.SYMBOL
                name = reference_id.removeprefix("symbol:")

            nodes.append(
                GraphNode(
                    id=scoped_id,
                    label=label,
                    name=name,
                    file_path=file_path,
                    language=language,
                    properties={
                        "repository": self.repository_name,
                        "reference": True,
                    },
                )
            )
            seen_node_ids.add(scoped_id)

        for pf in parsed_files:
            # File node
            file_node_id = self._scoped_id(f"file:{pf.file_path}")

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

            # Class and function nodes
            for entity in pf.entities:
                scoped_id = self._scoped_id(entity.id)

                if scoped_id in seen_node_ids:
                    continue

                node_label = (
                    NodeType.CLASS
                    if entity.type == "class"
                    else NodeType.FUNCTION
                )

                nodes.append(
                    GraphNode(
                        id=scoped_id,
                        label=node_label,
                        name=entity.name,
                        file_path=entity.file_path,
                        language=pf.language,
                        start_line=entity.start_line,
                        end_line=entity.end_line,
                        properties={"repository": self.repository_name},
                    )
                )
                seen_node_ids.add(scoped_id)

            # Relationships
            for rel in pf.relationships:
                source_id = (
                    f"file:{rel.source_id}"
                    if rel.source_id == pf.file_path
                    else rel.source_id
                )

                if (
                    rel.target_id.startswith("module:")
                    or rel.target_id.startswith("symbol:")
                ):
                    add_reference_node(
                        rel.target_id,
                        pf.file_path,
                        pf.language,
                    )

                edges.append(
                    GraphEdge(
                        source_id=self._scoped_id(source_id),
                        target_id=self._scoped_id(rel.target_id),
                        type=RelationType(rel.type),
                    )
                )

        return nodes, edges
     
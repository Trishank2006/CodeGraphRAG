
import re
from typing import Any, Dict, List, Optional

from graph.neo4j_store import Neo4jStore


class GraphQueries:
    """Repository-aware graph queries supporting legacy and scoped IDs."""

    def __init__(self, store: Neo4jStore):
        self.store = store

    @staticmethod
    def _scoped_id(repository: Optional[str], node_id: str) -> str:
        """Return a repository-scoped ID unless it is already scoped."""
        if not repository or node_id.startswith("repo:"):
            return node_id
        return f"repo:{repository}:{node_id}"

    @staticmethod
    def _file_id(file_path: str, repository: Optional[str]) -> str:
        """Normalize a file path to the graph's file-node ID format."""
        normalized = file_path.replace("\\", "/")
        if not normalized.startswith("file:"):
            normalized = f"file:{normalized}"
        return GraphQueries._scoped_id(repository, normalized)

    @staticmethod
    def _normalized_file_path(file_path: str) -> str:
        """Normalize separators and remove the optional file: prefix."""
        normalized = file_path.replace("\\", "/")
        if normalized.startswith("file:"):
            normalized = normalized[len("file:"):]
        return normalized.lstrip("/")

    @staticmethod
    def _node_return(prefix: str) -> str:
        """Build a consistent projection for a returned graph node."""
        return f"""
            {prefix}.id AS id,
            {prefix}.name AS name,
            {prefix}.label AS label,
            {prefix}.file_path AS file_path,
            {prefix}.language AS language,
            {prefix}.start_line AS start_line,
            {prefix}.end_line AS end_line,
            coalesce({prefix}.repository, 'default_repo') AS repository
        """

    def find_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        language: Optional[str] = None,
        repository: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find entities by ID, exact name, or partial name."""
        tokens = [
            token.lower()
            for token in re.split(r"[\s._/]+", query)
            if len(token) > 2
        ]
        scoped_query = self._scoped_id(repository, query)

        cypher = f"""
        MATCH (n:Node)
        WHERE (
            n.name = $query
            OR n.id = $query
            OR n.id = $scoped_query
            OR toLower(n.name) CONTAINS toLower($query)
            OR any(token IN $tokens WHERE toLower(n.name) CONTAINS token)
        )
        AND ($entity_type IS NULL OR toLower(n.label) = toLower($entity_type))
        AND ($language IS NULL OR toLower(n.language) = toLower($language))
        AND (
            $repository IS NULL
            OR n.repository = $repository
        )
        RETURN {self._node_return("n")}
        ORDER BY
            CASE
                WHEN n.id = $query OR n.id = $scoped_query THEN 0
                WHEN n.name = $query THEN 1
                ELSE 2
            END
        LIMIT 10
        """
        return self.store.execute_query(
            cypher,
            {
                "query": query,
                "scoped_query": scoped_query,
                "tokens": tokens,
                "entity_type": entity_type,
                "language": language,
                "repository": repository,
            },
        )

    def _get_file_relationship(
        self,
        file_path: str,
        relationship_type: str,
        target_variable: str,
        repository: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Query file relationships using legacy or scoped file IDs."""
        normalized_path = self._normalized_file_path(file_path)
        legacy_file_id = f"file:{normalized_path}"
        scoped_file_id = self._file_id(normalized_path, repository)

        cypher = f"""
        MATCH (f:Node)-[r:RELATION]->({target_variable}:Node)
        WHERE r.type = $relationship_type
          AND f.label = 'File'
          AND (
              f.id = $legacy_file_id
              OR f.id = $scoped_file_id
              OR f.name = $normalized_path
              OR f.file_path = $normalized_path
          )
          AND (
              $repository IS NULL
              OR (
                  f.repository = $repository
                  AND {target_variable}.repository = $repository
              )
          )
        RETURN {self._node_return(target_variable)}
        """
        return self.store.execute_query(
            cypher,
            {
                "relationship_type": relationship_type,
                "legacy_file_id": legacy_file_id,
                "scoped_file_id": scoped_file_id,
                "normalized_path": normalized_path,
                "repository": repository,
            },
        )

    def get_file_contains(
        self, file_path: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return nodes contained in a file."""
        return self._get_file_relationship(
            file_path, "CONTAINS", "entity", repository
        )

    def get_function_callers(
        self, function_name_or_id: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return functions that call the specified function."""
        target_id = self._scoped_id(repository, function_name_or_id)

        cypher = f"""
        MATCH (caller:Node)-[r:RELATION]->(target:Node)
        WHERE r.type = 'CALLS'
          AND (
              target.name = $name
              OR target.id = $name
              OR target.id = $target_id
          )
          AND (
              $repository IS NULL
              OR (
                  caller.repository = $repository
                  AND target.repository = $repository
              )
          )
        RETURN {self._node_return("caller")}
        """
        return self.store.execute_query(
            cypher,
            {
                "name": function_name_or_id,
                "target_id": target_id,
                "repository": repository,
            },
        )

    def get_function_callees(
        self, function_name_or_id: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return functions called by the specified function."""
        source_id = self._scoped_id(repository, function_name_or_id)

        cypher = f"""
        MATCH (source:Node)-[r:RELATION]->(callee:Node)
        WHERE r.type = 'CALLS'
          AND (
              source.name = $name
              OR source.id = $name
              OR source.id = $source_id
          )
          AND (
              $repository IS NULL
              OR (
                  source.repository = $repository
                  AND callee.repository = $repository
              )
          )
        RETURN {self._node_return("callee")}
        """
        return self.store.execute_query(
            cypher,
            {
                "name": function_name_or_id,
                "source_id": source_id,
                "repository": repository,
            },
        )

    def get_file_dependencies(
        self, file_path: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return imported dependencies for a file."""
        return self._get_file_relationship(
            file_path, "IMPORTS", "dep", repository
        )

    def get_dependencies(
        self, symbol_or_file: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find imports from a file or symbol without crossing repositories."""
        normalized_path = self._normalized_file_path(symbol_or_file)
        source_id = self._scoped_id(repository, symbol_or_file)
        legacy_file_id = f"file:{normalized_path}"
        scoped_file_id = self._file_id(normalized_path, repository)

        cypher = f"""
        MATCH (source:Node)-[r:RELATION]->(dep:Node)
        WHERE r.type = 'IMPORTS'
          AND (
              source.name = $name
              OR source.id = $source_id
              OR source.id = $legacy_file_id
              OR source.id = $scoped_file_id
          )
          AND (
              $repository IS NULL
              OR (
                  source.repository = $repository
                  AND dep.repository = $repository
              )
          )
        RETURN {self._node_return("dep")}
        """
        return self.store.execute_query(
            cypher,
            {
                "name": symbol_or_file,
                "source_id": source_id,
                "legacy_file_id": legacy_file_id,
                "scoped_file_id": scoped_file_id,
                "repository": repository,
            },
        )

    def get_class_methods(
        self, class_name: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return methods and functions contained in a class."""
        cypher = f"""
        MATCH (c:Node)-[r:RELATION]->(m:Node)
        WHERE c.label = 'Class'
          AND r.type = 'CONTAINS'
          AND c.name = $class_name
          AND m.label IN ['Function', 'Method']
          AND (
              $repository IS NULL
              OR (
                  c.repository = $repository
                  AND m.repository = $repository
              )
          )
        RETURN {self._node_return("m")}
        """
        return self.store.execute_query(
            cypher,
            {"class_name": class_name, "repository": repository},
        )

    def get_parent_classes(
        self, class_name_or_id: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return parent classes for a class name or ID."""
        child_id = self._scoped_id(repository, class_name_or_id)

        cypher = f"""
        MATCH (child:Node)-[r:RELATION]->(parent:Node)
        WHERE r.type = 'INHERITS'
          AND (
              child.name = $name
              OR child.id = $name
              OR child.id = $child_id
          )
          AND (
              $repository IS NULL
              OR (
                  child.repository = $repository
                  AND parent.repository = $repository
              )
          )
        RETURN {self._node_return("parent")}
        """
        return self.store.execute_query(
            cypher,
            {
                "name": class_name_or_id,
                "child_id": child_id,
                "repository": repository,
            },
        )

    def get_file_symbols(
        self, file_path: str, repository: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return symbols contained in a file."""
        return self._get_file_relationship(
            file_path, "CONTAINS", "sym", repository
        )

    def get_neighbors_with_depth(
        self,
        start_id_or_name: str,
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
        repository: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return nearby nodes with distance and relationship information."""
        max_hops = min(max(max_hops, 1), 3)
        start_id = self._scoped_id(repository, start_id_or_name)

        cypher = f"""
        MATCH path = (start:Node)-[:RELATION*1..{max_hops}]-(neighbor:Node)
        WHERE (
            start.id = $val
            OR start.id = $start_id
            OR start.name = $val
        )
        AND (
            $repository IS NULL
            OR (
                start.repository = $repository
                AND neighbor.repository = $repository
            )
        )
        AND (
            $relationship_types IS NULL
            OR all(
                rel IN relationships(path)
                WHERE rel.type IN $relationship_types
            )
        )
        RETURN
            neighbor.id AS id,
            neighbor.name AS name,
            neighbor.label AS label,
            neighbor.file_path AS file_path,
            neighbor.language AS language,
            neighbor.start_line AS start_line,
            neighbor.end_line AS end_line,
            coalesce(neighbor.repository, 'default_repo') AS repository,
            min(length(path)) AS distance,
            last(relationships(path)).type AS relationship
        ORDER BY distance ASC
        """
        return self.store.execute_query(
            cypher,
            {
                "val": start_id_or_name,
                "start_id": start_id,
                "relationship_types": relationship_types,
                "repository": repository,
            },
        )

    def graph_search(
        self,
        entity: str,
        relation: str = "CALLS",
        depth: int = 1,
        repository: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search directed graph paths without crossing repository boundaries."""
        depth = min(max(depth, 1), 3)
        start_id = self._scoped_id(repository, entity)

        cypher = f"""
        MATCH path = (start:Node)-[:RELATION*1..{depth}]->(target:Node)
        WHERE (
            start.name = $entity
            OR start.id = $entity
            OR start.id = $start_id
        )
        AND all(rel IN relationships(path) WHERE rel.type = $relation)
        AND (
            $repository IS NULL
            OR all(n IN nodes(path) WHERE n.repository = $repository)
        )
        RETURN
            [n IN nodes(path) |
                {{id: n.id, name: n.name, file_path: n.file_path}}
            ] AS chain,
            length(path) AS depth
        """
        return self.store.execute_query(
            cypher,
            {
                "entity": entity,
                "start_id": start_id,
                "relation": relation,
                "repository": repository,
            },
        )

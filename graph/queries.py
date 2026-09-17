import re
from typing import List, Dict, Any, Optional
from graph.neo4j_store import Neo4jStore


class GraphQueries:
    def __init__(self, store: Neo4jStore):
        self.store = store

    def find_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-strategy entity lookup: exact ID, exact name, symbol tokens, and substring matches."""
        tokens = [t.lower() for t in re.split(r"[\s._/]+", query) if len(t) > 2]

        cypher = """
        MATCH (n:Node)
        WHERE (
            n.name = $query 
            OR n.id = $query 
            OR toLower(n.name) CONTAINS toLower($query)
            OR any(token IN $tokens WHERE toLower(n.name) CONTAINS token)
        )
        AND ($entity_type IS NULL OR toLower(n.label) = toLower($entity_type))
        AND ($language IS NULL OR toLower(n.language) = toLower($language))
        RETURN n.id AS id, n.name AS name, n.label AS label,
               n.file_path AS file_path, n.language AS language,
               n.start_line AS start_line, n.end_line AS end_line,
               coalesce(n.repository, 'default_repo') AS repository
        LIMIT 10
        """
        return self.store.execute_query(
            cypher,
            {
                "query": query,
                "tokens": tokens,
                "entity_type": entity_type,
                "language": language,
            },
        )

    def get_file_contains(self, file_path: str) -> List[Dict[str, Any]]:
        file_id = f"file:{file_path}" if not file_path.startswith("file:") else file_path
        cypher = """
        MATCH (f {id: $file_id})-[r:RELATION {type: 'CONTAINS'}]->(entity)
        RETURN entity.id AS id, entity.name AS name, entity.label AS label,
               entity.file_path AS file_path, entity.language AS language,
               entity.start_line AS start_line, entity.end_line AS end_line,
               coalesce(entity.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"file_id": file_id})

    def get_function_callers(self, function_name_or_id: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (caller)-[r:RELATION {type: 'CALLS'}]->(target)
        WHERE target.name = $name OR target.id = $name OR target.id = 'symbol:' + $name
        RETURN caller.id AS id, caller.name AS name, caller.label AS label,
               caller.file_path AS file_path, caller.language AS language,
               caller.start_line AS start_line, caller.end_line AS end_line,
               coalesce(caller.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"name": function_name_or_id})

    def get_function_callees(self, function_name_or_id: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (source)-[r:RELATION {type: 'CALLS'}]->(callee)
        WHERE source.name = $name OR source.id = $name
        RETURN callee.id AS id, callee.name AS name, callee.label AS label,
               callee.file_path AS file_path, callee.language AS language,
               callee.start_line AS start_line, callee.end_line AS end_line,
               coalesce(callee.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"name": function_name_or_id})

    def get_file_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        file_id = f"file:{file_path}" if not file_path.startswith("file:") else file_path
        cypher = """
        MATCH (f {id: $file_id})-[r:RELATION {type: 'IMPORTS'}]->(dep)
        RETURN dep.id AS id, dep.name AS name, dep.label AS label,
               dep.file_path AS file_path, dep.language AS language,
               dep.start_line AS start_line, dep.end_line AS end_line,
               coalesce(dep.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"file_id": file_id})

    def get_dependencies(self, symbol_or_file: str) -> List[Dict[str, Any]]:
        """Resolves dependencies by file path or symbol."""
        cypher = """
        MATCH (source)-[r:RELATION {type: 'IMPORTS'}]->(dep)
        WHERE source.name = $name OR source.id = $name OR source.id = 'file:' + $name
        RETURN dep.id AS id, dep.name AS name, dep.label AS label,
               dep.file_path AS file_path, dep.language AS language,
               dep.start_line AS start_line, dep.end_line AS end_line,
               coalesce(dep.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"name": symbol_or_file})

    def get_class_methods(self, class_name: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (c:Node {label: 'Class', name: $class_name})-[r:RELATION {type: 'CONTAINS'}]->(m:Node)
        WHERE m.label IN ['Function', 'Method']
        RETURN m.id AS id, m.name AS name, m.label AS label,
               m.file_path AS file_path, m.language AS language,
               m.start_line AS start_line, m.end_line AS end_line,
               coalesce(m.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"class_name": class_name})

    def get_parent_classes(self, class_name_or_id: str) -> List[Dict[str, Any]]:
        """Retrieves inherited parent classes or interfaces."""
        cypher = """
        MATCH (child)-[r:RELATION {type: 'INHERITS'}]->(parent)
        WHERE child.name = $name OR child.id = $name
        RETURN parent.id AS id, parent.name AS name, parent.label AS label,
               parent.file_path AS file_path, parent.language AS language,
               parent.start_line AS start_line, parent.end_line AS end_line,
               coalesce(parent.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"name": class_name_or_id})

    def get_file_symbols(self, file_path: str) -> List[Dict[str, Any]]:
        """Finds all symbol nodes declared within a specific file."""
        file_id = f"file:{file_path}" if not file_path.startswith("file:") else file_path
        cypher = """
        MATCH (f {id: $file_id})-[r:RELATION {type: 'CONTAINS'}]->(sym:Node)
        RETURN sym.id AS id, sym.name AS name, sym.label AS label,
               sym.file_path AS file_path, sym.language AS language,
               sym.start_line AS start_line, sym.end_line AS end_line,
               coalesce(sym.repository, 'default_repo') AS repository
        """
        return self.store.execute_query(cypher, {"file_id": file_id})

    def get_neighbors_with_depth(
        self,
        start_id_or_name: str,
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves neighboring nodes up to max_hops with edge types and shortest distance."""
        max_hops = min(max(max_hops, 1), 3)
        cypher = f"""
        MATCH path = (start)-[r:RELATION*1..{max_hops}]-(neighbor:Node)
        WHERE start.id = $val OR start.name = $val
        RETURN DISTINCT neighbor.id AS id, neighbor.name AS name, neighbor.label AS label,
               neighbor.file_path AS file_path, neighbor.language AS language,
               neighbor.start_line AS start_line, neighbor.end_line AS end_line,
               coalesce(neighbor.repository, 'default_repo') AS repository,
               min(length(path)) AS distance,
               last(relationships(path)).type AS relationship
        ORDER BY distance ASC
        """
        return self.store.execute_query(cypher, {"val": start_id_or_name})

    def graph_search(
        self,
        entity: str,
        relation: str = "CALLS",
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Public Graph Search API hiding Cypher logic from the retrieval layer."""
        depth = min(max(depth, 1), 3)
        cypher = f"""
        MATCH path = (start {{name: $entity}})-[r:RELATION*1..{depth} {{type: $relation}}]->(target)
        RETURN [n in nodes(path) | {{id: n.id, name: n.name, file_path: n.file_path}}] AS chain,
               length(path) AS depth
        """
        return self.store.execute_query(cypher, {"entity": entity, "relation": relation})
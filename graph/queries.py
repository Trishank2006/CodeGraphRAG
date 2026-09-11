from typing import List, Dict, Any, Optional
from graph.neo4j_store import Neo4jStore


class GraphQueries:
    def __init__(self, store: Neo4jStore):
        self.store = store

    def get_file_contains(self, file_path: str) -> List[Dict[str, Any]]:
        file_id = f"file:{file_path}"
        cypher = """
        MATCH (f {id: $file_id})-[r:RELATION {type: 'CONTAINS'}]->(entity)
        RETURN entity.id AS id, entity.name AS name, entity.label AS type,
               entity.start_line AS start_line, entity.end_line AS end_line
        """
        return self.store.execute_query(cypher, {"file_id": file_id})

    def get_function_callers(self, function_name: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (caller)-[r:RELATION {type: 'CALLS'}]->(target {name: $function_name})
        RETURN caller.id AS id, caller.name AS name, caller.file_path AS file_path
        """
        return self.store.execute_query(cypher, {"function_name": function_name})

    def get_function_callees(self, function_name: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (source {name: $function_name})-[r:RELATION {type: 'CALLS'}]->(callee)
        RETURN callee.id AS id, callee.name AS name, callee.file_path AS file_path
        """
        return self.store.execute_query(cypher, {"function_name": function_name})

    def get_file_dependencies(self, file_path: str) -> List[Dict[str, Any]]:
        file_id = f"file:{file_path}"
        cypher = """
        MATCH (f {id: $file_id})-[r:RELATION {type: 'IMPORTS'}]->(dep)
        RETURN dep.id AS id, dep.name AS name, dep.file_path AS file_path
        """
        return self.store.execute_query(cypher, {"file_id": file_id})

    def get_dependencies(self, symbol_or_file: str) -> List[Dict[str, Any]]:
        """Unified dependency resolver for classes, functions, or files."""
        cypher = """
        MATCH (source {name: $name})-[r:RELATION {type: 'IMPORTS'}]->(dep)
        RETURN dep.id AS id, dep.name AS name, dep.file_path AS file_path
        UNION
        MATCH (source {id: $name})-[r:RELATION {type: 'IMPORTS'}]->(dep)
        RETURN dep.id AS id, dep.name AS name, dep.file_path AS file_path
        """
        return self.store.execute_query(cypher, {"name": symbol_or_file})

    def get_class_methods(self, class_name: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (c:Node {label: 'Class', name: $class_name})-[r:RELATION {type: 'CONTAINS'}]->(m:Node)
        WHERE m.label IN ['Function', 'Method']
        RETURN m.id AS id, m.name AS name, m.start_line AS start_line, m.end_line AS end_line
        """
        return self.store.execute_query(cypher, {"class_name": class_name})

    def graph_search(
        self,
        entity: str,
        relation: str = "CALLS",
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Public Graph Search API hiding all Cypher logic from retrieval layer."""
        depth = min(max(depth, 1), 3)  # Bound depth between 1 and 3 hops
        cypher = f"""
        MATCH path = (start {{name: $entity}})-[r:RELATION*1..{depth} {{type: $relation}}]->(target)
        RETURN [n in nodes(path) | {{id: n.id, name: n.name, file_path: n.file_path}}] AS chain,
               length(path) AS depth
        """
        return self.store.execute_query(cypher, {"entity": entity, "relation": relation})
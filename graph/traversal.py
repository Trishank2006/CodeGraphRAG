from typing import List, Dict, Any
from graph.neo4j_store import Neo4jStore


class GraphTraversal:
    def __init__(self, store: Neo4jStore):
        self.store = store

    def get_call_chain(
        self, start_symbol: str, depth: int = 2
    ) -> List[Dict[str, Any]]:
        if depth < 1:
            return []

        # Parameterized depth traversal
        cypher = f"""
        MATCH path = (start {{name: $symbol}})-[r:RELATION*1..{depth} {{type: 'CALLS'}}]->(target)
        RETURN [n in nodes(path) | {{id: n.id, name: n.name, file_path: n.file_path}}] AS chain,
               length(path) AS depth
        ORDER BY depth ASC
        """
        results = self.store.execute_query(cypher, {"symbol": start_symbol})
        return results
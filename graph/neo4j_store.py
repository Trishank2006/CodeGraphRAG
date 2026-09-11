from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from graph.models import GraphNode, GraphEdge


class Neo4jStore:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        driver: Optional[Driver] = None,
    ):
        self.driver = driver or GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def create_constraints(self):
        queries = [
            "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT func_id_unique IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT class_id_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    def insert_nodes(self, nodes: List[GraphNode]):
        query = """
        UNWIND $batch AS data
        MERGE (n:Node {id: data.id})
        SET n.name = data.name,
            n.file_path = data.file_path,
            n.language = data.language,
            n.start_line = data.start_line,
            n.end_line = data.end_line
        WITH n, data
        CALL apoc.create.addLabels(n, [data.label]) YIELD node
        RETURN count(node)
        """
        # Fallback query without APOC plugin for maximum cross-environment portability
        portable_query = """
        UNWIND $batch AS data
        MERGE (n {id: data.id})
        SET n.name = data.name,
            n.file_path = data.file_path,
            n.language = data.language,
            n.start_line = data.start_line,
            n.end_line = data.end_line,
            n.label = data.label
        """
        payload = [node.model_dump() for node in nodes]
        with self.driver.session() as session:
            session.run(portable_query, batch=payload)

    def insert_edges(self, edges: List[GraphEdge]):
        query = """
        UNWIND $batch AS data
        MATCH (source {id: data.source_id})
        MATCH (target {id: data.target_id})
        MERGE (source)-[r:RELATION {type: data.type}]->(target)
        """
        payload = [edge.model_dump() for edge in edges]
        with self.driver.session() as session:
            session.run(query, batch=payload)
from typing import Dict, Any
from graph.neo4j_store import Neo4jStore


def check_graph_health(store: Neo4jStore | None = None) -> Dict[str, Any]:
    """Inspects connectivity and database statistics for Neo4j."""
    store = store or Neo4jStore()
    try:
        ping_res = store.execute_query("RETURN 1 AS ping")
        if not ping_res or ping_res[0].get("ping") != 1:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Ping query returned unexpected result",
            }

        count_res = store.execute_query(
            "MATCH (n) OPTIONAL MATCH ()-[r]->() RETURN count(DISTINCT n) AS node_count, count(DISTINCT r) AS rel_count"
        )
        node_count = count_res[0].get("node_count", 0) if count_res else 0
        rel_count = count_res[0].get("rel_count", 0) if count_res else 0

        return {
            "status": "healthy",
            "connected": True,
            "database": store.database or "neo4j",
            "node_count": node_count,
            "relationship_count": rel_count,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }
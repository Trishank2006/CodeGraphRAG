from __future__ import annotations

from typing import Any

from graph.neo4j_store import Neo4jStore


def check_graph_health(store: Neo4jStore | None = None) -> dict[str, Any]:
    """Check Neo4j connectivity and report basic graph statistics.

    Never raises: failures are reported as ``status: unhealthy``. A store
    created here is closed again; a store passed in is left open.
    """
    owns_store = store is None

    try:
        store = store or Neo4jStore()

        ping = store.execute_query("RETURN 1 AS ping")
        if not ping or ping[0].get("ping") != 1:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Ping query returned an unexpected result",
            }

        # Two separate counts are answered from Neo4j's count store. A single
        # MATCH over nodes and relationships would build their full product.
        nodes = store.execute_query("MATCH (n) RETURN count(n) AS node_count")
        relationships = store.execute_query(
            "MATCH ()-[r]->() RETURN count(r) AS rel_count"
        )

        return {
            "status": "healthy",
            "connected": True,
            "database": getattr(store, "database", None) or "neo4j",
            "node_count": nodes[0].get("node_count", 0) if nodes else 0,
            "relationship_count": (
                relationships[0].get("rel_count", 0) if relationships else 0
            ),
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(exc),
        }
    finally:
        if owns_store and store is not None:
            store.close()

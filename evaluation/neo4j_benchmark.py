import time
from typing import Dict, Any, List, Callable
from evaluation.latency import percentile


def benchmark_neo4j_query(query_func: Callable[[], Any], iterations: int = 10) -> Dict[str, float]:
    """Measures P50, P95, and P99 query latency for a Neo4j operation."""
    latencies: List[float] = []
    for _ in range(max(1, iterations)):
        t0 = time.perf_counter()
        query_func()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    return {
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
    }


def calculate_indexing_throughput(
    total_files: int,
    total_nodes: int,
    total_edges: int,
    duration_seconds: float,
) -> Dict[str, float]:
    """Computes indexing throughput metrics for AST-to-Neo4j persistence."""
    duration = max(duration_seconds, 0.001)
    return {
        "duration_seconds": round(duration, 3),
        "files_per_second": round(total_files / duration, 2),
        "nodes_per_second": round(total_nodes / duration, 2),
        "edges_per_second": round(total_edges / duration, 2),
    }
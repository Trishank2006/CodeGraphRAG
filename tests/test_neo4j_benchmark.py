from evaluation.neo4j_benchmark import benchmark_neo4j_query, calculate_indexing_throughput


def test_neo4j_query_latency():
    def dummy_query():
        return 1

    res = benchmark_neo4j_query(dummy_query, iterations=5)
    assert "p50_ms" in res
    assert "p95_ms" in res
    assert "p99_ms" in res
    assert res["p50_ms"] >= 0.0


def test_indexing_throughput():
    res = calculate_indexing_throughput(
        total_files=10, total_nodes=100, total_edges=200, duration_seconds=2.0
    )
    assert res["files_per_second"] == 5.0
    assert res["nodes_per_second"] == 50.0
    assert res["edges_per_second"] == 100.0
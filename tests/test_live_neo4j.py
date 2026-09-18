import os
import pytest
from graph.neo4j_store import Neo4jStore
from graph.builder import GraphBuilder
from graph.retrieval import GraphRetriever
from graph.grounding import GraphGroundingService
from parser.models import CodeEntity


@pytest.mark.skipif(
    os.getenv("CODEGRAPHRAG_RUN_LIVE_TESTS") != "1",
    reason="Live tests require an active Neo4j instance and CODEGRAPHRAG_RUN_LIVE_TESTS=1",
)
def test_live_neo4j_pipeline():
    """End-to-end integration test validating CodeEntity -> Builder -> Neo4j -> Retriever -> Grounding."""
    store = Neo4jStore()
    builder = GraphBuilder()
    retriever = GraphRetriever(store)
    grounding = GraphGroundingService(store)

    entities = [
        CodeEntity(
            id="auth_live.py:class:AuthService",
            type="class",
            name="AuthService",
            file_path="auth_live.py",
            start_line=1,
            end_line=10,
        ),
        CodeEntity(
            id="auth_live.py:function:login",
            type="function",
            name="login",
            file_path="auth_live.py",
            start_line=2,
            end_line=5,
        ),
    ]

    nodes, edges = builder.build(entities, repository="live_test_repo")
    store.insert_nodes(nodes)
    store.insert_edges(edges)

    # Validate retrieval against live database
    results = retriever.search("login", repository="live_test_repo")
    assert len(results) > 0

    # Validate structural grounding extraction
    evidence = grounding.get_evidence("login")
    assert any(e.source_entity == "login" or e.target_entity == "login" for e in evidence)

    # Cleanup test data
    store.execute_query("MATCH (n:Node {repository: 'live_test_repo'}) DETACH DELETE n")
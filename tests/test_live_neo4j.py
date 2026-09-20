import os
import pytest
from graph.neo4j_store import Neo4jStore
from graph.builder import GraphBuilder
from graph.retrieval import GraphRetriever
from graph.grounding import GraphGroundingService
from parser.models import ParsedFile, CodeEntity, CodeRelationship


@pytest.mark.skipif(
    os.getenv("CODEGRAPHRAG_RUN_LIVE_TESTS") != "1",
    reason="Live tests require an active Neo4j instance and CODEGRAPHRAG_RUN_LIVE_TESTS=1",
)
def test_live_neo4j_pipeline():
    """End-to-end integration test validating CodeEntity -> Builder -> Neo4j -> Retriever -> Grounding."""
    store = Neo4jStore()
    builder = GraphBuilder(repository_name="live_test_repo")
    retriever = GraphRetriever(store)
    grounding = GraphGroundingService(store)

    auth_class = CodeEntity(
        id="auth_live.py:class:AuthService",
        type="class",
        name="AuthService",
        file_path="auth_live.py",
        start_line=1,
        end_line=10,
    )
    login_func = CodeEntity(
        id="auth_live.py:function:login",
        type="function",
        name="login",
        file_path="auth_live.py",
        start_line=2,
        end_line=5,
    )
    rel = CodeRelationship(
        source_id=auth_class.id,
        target_id=login_func.id,
        type="CONTAINS",
    )

    parsed_file = ParsedFile(
        file_path="auth_live.py",
        language="python",
        entities=[auth_class, login_func],
        relationships=[rel],
    )

    nodes, edges = builder.build_from_parsed_files([parsed_file])

    try:
        store.insert_nodes(nodes)
        store.insert_edges(edges)

        # Validate retrieval against live database
        results = retriever.search("login", repository="live_test_repo")
        assert len(results) > 0
        assert any(getattr(r, "symbol", "") == "login" or "login" in getattr(r, "content", "") for r in results)

        # Validate structural grounding extraction
        evidence = grounding.get_evidence("login")
        assert isinstance(evidence, list)

    finally:
        # Cleanup test data using repository isolation
        store.execute_query("MATCH (n {repository: $repo}) DETACH DELETE n", {"repo": "live_test_repo"})
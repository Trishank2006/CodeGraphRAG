from unittest.mock import MagicMock

from graph.builder import GraphBuilder
from graph.models import GraphNode
from graph.neo4j_store import Neo4jStore
from graph.schema import NodeType
from parser.core_parser import parse_source_code


def test_graph_builder_creates_reference_nodes_for_calls_and_imports():
    parsed = parse_source_code(
        "service.py",
        "import external_lib\n\ndef run():\n    execute()\n",
        "python",
    )

    nodes, _ = GraphBuilder("repo").build_from_parsed_files([parsed])
    node_ids = {node.id for node in nodes}

    assert "module:external_lib" in node_ids
    assert "symbol:execute" in node_ids


def test_neo4j_store_uses_queryable_node_label_and_repository_metadata():
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = None
    driver = MagicMock()
    driver.session.return_value = session
    store = Neo4jStore(driver=driver)

    store.insert_nodes(
        [
            GraphNode(
                id="symbol:execute",
                label=NodeType.SYMBOL,
                name="execute",
                properties={"repository": "repo", "reference": True},
            )
        ]
    )

    query = session.run.call_args.args[0]
    payload = session.run.call_args.kwargs["batch"]
    assert "MERGE (n {id: data.id})" in query
    assert "SET n:Node" in query
    assert "n.repository" in query
    assert payload[0]["properties"]["repository"] == "repo"

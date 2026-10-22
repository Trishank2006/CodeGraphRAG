
from parser.models import ParsedFile, CodeEntity, CodeRelationship
from graph.builder import GraphBuilder
from graph.schema import NodeType, RelationType


def test_graph_builder_transforms_ast_to_nodes_and_edges():
    entity = CodeEntity(
        id="src/payment.py:function:process_payment",
        type="function",
        name="process_payment",
        file_path="src/payment.py",
        start_line=10,
        end_line=25,
    )
    rel = CodeRelationship(
        source_id="src/payment.py",
        target_id=entity.id,
        type="CONTAINS",
    )
    parsed_file = ParsedFile(
        file_path="src/payment.py",
        language="python",
        entities=[entity],
        relationships=[rel],
    )

    builder = GraphBuilder(repository_name="test_repo")
    nodes, edges = builder.build_from_parsed_files([parsed_file])

    assert len(nodes) == 2  # 1 File node, 1 Function node
    assert len(edges) == 1

    file_node = next(n for n in nodes if n.label == NodeType.FILE)
    func_node = next(n for n in nodes if n.label == NodeType.FUNCTION)

    # IDs are scoped to the repository to prevent cross-repository collisions.
    assert file_node.id == "repo:test_repo:file:src/payment.py"
    assert func_node.id == (
        "repo:test_repo:src/payment.py:function:process_payment"
    )
    assert func_node.name == "process_payment"
    assert func_node.start_line == 10

    assert edges[0].source_id == "repo:test_repo:file:src/payment.py"
    assert edges[0].target_id == (
        "repo:test_repo:src/payment.py:function:process_payment"
    )
    assert edges[0].type == RelationType.CONTAINS


def test_graph_builder_resolves_call_to_local_definition():
    caller = CodeEntity(
        id="src/a.py:function:caller",
        type="function",
        name="caller",
        file_path="src/a.py",
        start_line=1,
        end_line=5,
    )
    target = CodeEntity(
        id="src/a.py:function:search",
        type="function",
        name="search",
        file_path="src/a.py",
        start_line=7,
        end_line=10,
    )
    parsed_file = ParsedFile(
        file_path="src/a.py",
        language="python",
        entities=[caller, target],
        relationships=[
            CodeRelationship(
                source_id=caller.id,
                target_id="symbol:search",
                type="CALLS",
            )
        ],
    )

    nodes, edges = GraphBuilder("test_repo").build_from_parsed_files(
        [parsed_file]
    )

    assert any(
        edge.source_id.endswith("src/a.py:function:caller")
        and edge.target_id.endswith("src/a.py:function:search")
        and edge.type == RelationType.CALLS
        for edge in edges
    )
    assert not any(
        node.label == NodeType.SYMBOL and node.name == "search"
        for node in nodes
    )


def test_graph_builder_keeps_ambiguous_call_as_reference_node():
    caller = CodeEntity(
        id="src/caller.py:function:caller",
        type="function",
        name="caller",
        file_path="src/caller.py",
        start_line=1,
        end_line=5,
    )
    first = CodeEntity(
        id="src/a.py:function:search",
        type="function",
        name="search",
        file_path="src/a.py",
        start_line=1,
        end_line=2,
    )
    second = CodeEntity(
        id="src/b.py:function:search",
        type="function",
        name="search",
        file_path="src/b.py",
        start_line=1,
        end_line=2,
    )
    parsed_files = [
        ParsedFile(file_path="src/caller.py", language="python",
                   entities=[caller], relationships=[
                       CodeRelationship(source_id=caller.id,
                                       target_id="symbol:search", type="CALLS")
                   ]),
        ParsedFile(file_path="src/a.py", language="python",
                   entities=[first], relationships=[]),
        ParsedFile(file_path="src/b.py", language="python",
                   entities=[second], relationships=[]),
    ]

    nodes, edges = GraphBuilder("test_repo").build_from_parsed_files(
        parsed_files
    )

    assert any(
        node.label == NodeType.SYMBOL
        and node.name == "search"
        and node.properties.get("reference") is True
        for node in nodes
    )
    assert any(
        edge.target_id == "repo:test_repo:symbol:search"
        and edge.type == RelationType.CALLS
        for edge in edges
    )

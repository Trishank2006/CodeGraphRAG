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

    assert file_node.id == "file:src/payment.py"
    assert func_node.id == entity.id
    assert func_node.name == "process_payment"
    assert func_node.start_line == 10

    assert edges[0].source_id == "file:src/payment.py"
    assert edges[0].target_id == entity.id
    assert edges[0].type == RelationType.CONTAINS
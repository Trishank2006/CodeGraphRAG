from parser.core_parser import parse_source_code
from graph.builder import GraphBuilder
from graph.schema import RelationType


def test_person_2_definition_of_done():
    order_code = """
class OrderService:
    def create_order(self):
        payment_service.process_payment()
"""
    payment_code = """
class PaymentService:
    def process_payment(self):
        database.save()
"""
    parsed_order = parse_source_code("services/order.py", order_code, "python")
    parsed_payment = parse_source_code("services/payment.py", payment_code, "python")

    builder = GraphBuilder(repository_name="ecommerce")
    nodes, edges = builder.build_from_parsed_files([parsed_order, parsed_payment])

    node_names = {n.name for n in nodes}
    assert "OrderService" in node_names
    assert "create_order" in node_names
    assert "PaymentService" in node_names
    assert "process_payment" in node_names

    # Check structural CONTAINS
    contains_edges = [e for e in edges if e.type == RelationType.CONTAINS]
    assert len(contains_edges) >= 2

    # Check CALLS edges
    call_edges = [e for e in edges if e.type == RelationType.CALLS]
    call_targets = {e.target_id for e in call_edges}
    assert "symbol:process_payment" in call_targets
    assert "symbol:save" in call_targets
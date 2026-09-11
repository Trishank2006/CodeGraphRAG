from parser.core_parser import parse_source_code


def test_parse_python_imports_and_calls():
    code = """
import math
from os import path

def calculate(value):
    return math.sqrt(value)

def main():
    calculate(16)
"""
    parsed = parse_source_code("main.py", code, "python")

    # Verify entities
    entity_names = {e.name for e in parsed.entities}
    assert "calculate" in entity_names
    assert "main" in entity_names

    # Verify relationships
    types = [r.type for r in parsed.relationships]
    assert "CONTAINS" in types
    assert "IMPORTS" in types
    assert "CALLS" in types

    # Verify import target
    import_targets = [r.target_id for r in parsed.relationships if r.type == "IMPORTS"]
    assert "module:math" in import_targets
    assert "module:os" in import_targets

    # Verify call edge from main -> calculate
    call_edges = [r for r in parsed.relationships if r.type == "CALLS"]
    assert any(r.target_id == "symbol:calculate" for r in call_edges)
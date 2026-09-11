from parser.core_parser import parse_source_code


def test_file_import_relationships():
    code = """
import numpy as np
import pandas as pd
from os import environ
from services.auth import AuthService
"""
    parsed = parse_source_code("src/app.py", code, "python")
    import_targets = [r.target_id for r in parsed.relationships if r.type == "IMPORTS"]

    assert "module:numpy" in import_targets
    assert "module:pandas" in import_targets
    assert "module:os" in import_targets
    assert "module:services.auth" in import_targets
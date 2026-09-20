from graph.evidence import GraphEvidence
from evaluation.citation_validation import GraphCitationValidator


def test_citation_3way_alignment():
    ev = GraphEvidence(
        source_entity="login",
        relationship="CALLS",
        target_entity="validate",
        source_file="auth.py",
        target_file="auth.py",
        source_start_line=2,
        source_end_line=4,
        target_start_line=10,
        target_end_line=12,
    )
    source_code = "import os\ndef login():\n    return True\n"

    res = GraphCitationValidator.validate_3way_alignment(
        ev, file_content=source_code, ast_start_line=2, ast_end_line=4
    )
    assert res["is_valid"] is True
    assert res["ast_aligned"] is True
    assert res["source_aligned"] is True
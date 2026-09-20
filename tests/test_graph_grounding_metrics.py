from graph.evidence import GraphEvidence, PathStep, GraphPath, EntityMetadata
from evaluation.graph_grounding_metrics import (
    evaluate_evidence_accuracy,
    evaluate_graph_path_coverage,
)


def test_evidence_accuracy():
    ev = GraphEvidence(
        source_entity="login",
        relationship="CALLS",
        target_entity="validate_token",
        source_file="auth.py",
        target_file="auth.py",
        source_start_line=10,
        source_end_line=15,
        target_start_line=20,
        target_end_line=25,
    )
    expected = [{"source": "login", "relation": "CALLS", "target": "validate_token"}]
    res = evaluate_evidence_accuracy([ev], expected)
    assert res["accuracy"] == 1.0
    assert res["line_preservation_rate"] == 1.0


def test_graph_path_coverage():
    src = EntityMetadata(id="1", name="login", label="Function", file_path="auth.py")
    tgt = EntityMetadata(id="2", name="validate", label="Function", file_path="auth.py")
    step = PathStep(
        source=src,
        relationship="CALLS",
        target=tgt,
    )
    path = GraphPath(source=src, target=tgt, steps=[step])
    expected = [{"source": "login", "relation": "CALLS", "target": "validate"}]
    res = evaluate_graph_path_coverage(path, expected)
    assert res["path_found"] is True
    assert res["hop_coverage"] == 1.0
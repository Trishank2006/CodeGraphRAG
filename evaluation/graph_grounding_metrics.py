from typing import List, Dict, Any
from graph.evidence import GraphEvidence, GraphPath


def evaluate_evidence_accuracy(
    retrieved_evidence: List[GraphEvidence],
    expected_evidence: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Validates whether extracted GraphEvidence items match expected structural connections.
    Checks source entity, relationship, target entity, and line preservation.
    """
    if not expected_evidence:
        return {
            "evidence_count": len(retrieved_evidence),
            "accuracy": 1.0 if not retrieved_evidence else 0.0,
            "line_preservation_rate": 1.0,
        }

    if not retrieved_evidence:
        return {
            "evidence_count": 0,
            "accuracy": 0.0,
            "line_preservation_rate": 0.0,
        }

    matches = 0
    valid_lines = 0

    for ev in retrieved_evidence:
        if ev.source_start_line > 0 and ev.target_start_line > 0:
            valid_lines += 1

        rel = getattr(ev, "relationship", getattr(ev, "relation_type", None))
        for exp in expected_evidence:
            if (
                ev.source_entity == exp.get("source")
                and rel == exp.get("relation")
                and ev.target_entity == exp.get("target")
            ):
                matches += 1
                break

    accuracy = float(matches / len(expected_evidence))
    line_preservation_rate = float(valid_lines / len(retrieved_evidence))

    return {
        "evidence_count": len(retrieved_evidence),
        "accuracy": min(accuracy, 1.0),
        "line_preservation_rate": line_preservation_rate,
    }


def evaluate_graph_path_coverage(
    retrieved_path: GraphPath | None,
    expected_hops: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Evaluates whether GraphPath hops and step directions match ground truth."""
    if not expected_hops:
        return {"path_found": retrieved_path is not None, "hop_coverage": 1.0}

    if not retrieved_path or not getattr(retrieved_path, "steps", None):
        return {"path_found": False, "hop_coverage": 0.0}

    matched_hops = 0
    for step in retrieved_path.steps:
        for exp in expected_hops:
            if (
                step.source.name == exp.get("source")
                and step.relationship == exp.get("relation")
                and step.target.name == exp.get("target")
            ):
                matched_hops += 1
                break

    return {
        "path_found": True,
        "hop_count": len(retrieved_path.steps),
        "hop_coverage": float(matched_hops / len(expected_hops)),
    }
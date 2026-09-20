from typing import List, Dict, Any


def evaluate_path_reconstruction(
    retrieved_path: List[str],
    expected_path: List[str],
) -> Dict[str, Any]:
    """
    Evaluates shortest execution path correctness.
    Handles exact sequence match, node accuracy, length difference, and missing path handling.
    """
    if not expected_path:
        return {
            "exact_match": len(retrieved_path) == 0,
            "node_accuracy": 1.0 if not retrieved_path else 0.0,
            "length_difference": len(retrieved_path),
            "valid_path": True,
        }

    if not retrieved_path:
        return {
            "exact_match": False,
            "node_accuracy": 0.0,
            "length_difference": len(expected_path),
            "valid_path": False,
        }

    exact_match = retrieved_path == expected_path

    expected_set = set(expected_path)
    retrieved_set = set(retrieved_path)
    overlap = len(retrieved_set.intersection(expected_set))
    node_accuracy = float(overlap / len(expected_set))
    length_diff = abs(len(retrieved_path) - len(expected_path))

    return {
        "exact_match": exact_match,
        "node_accuracy": node_accuracy,
        "length_difference": length_diff,
        "valid_path": True,
    }
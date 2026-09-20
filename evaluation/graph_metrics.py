from typing import List, Dict, Any, Set, Tuple


def calculate_graph_hit_rate_at_k(
    retrieved_entities: List[str],
    expected_entities: List[str],
    k: int = 10,
) -> float:
    """Calculates whether at least one expected entity is present in top K."""
    if not expected_entities:
        return 0.0
    top_k = retrieved_entities[:k]
    expected_set = set(expected_entities)
    return 1.0 if any(e in expected_set for e in top_k) else 0.0


def calculate_graph_recall_at_k(
    retrieved_entities: List[str],
    expected_entities: List[str],
    k: int = 10,
) -> float:
    """Calculates the proportion of expected entities retrieved in top K."""
    if not expected_entities:
        return 0.0
    top_k = set(retrieved_entities[:k])
    expected_set = set(expected_entities)
    found = len(top_k.intersection(expected_set))
    return float(found / len(expected_set))


def calculate_graph_mrr(
    retrieved_entities: List[str],
    expected_entities: List[str],
) -> float:
    """Calculates Mean Reciprocal Rank for the first relevant graph entity."""
    if not expected_entities:
        return 0.0
    expected_set = set(expected_entities)
    for rank, entity in enumerate(retrieved_entities, start=1):
        if entity in expected_set:
            return 1.0 / rank
    return 0.0


def calculate_relationship_accuracy(
    retrieved_edges: List[Tuple[str, str, str]],
    expected_edges: List[Tuple[str, str, str]],
) -> float:
    """
    Calculates relationship type accuracy.
    Edges are represented as (source, relation_type, target).
    """
    if not expected_edges:
        return 0.0
    expected_set = set(expected_edges)
    retrieved_set = set(retrieved_edges)
    correct = len(retrieved_set.intersection(expected_set))
    return float(correct / len(expected_set))
from typing import Dict, Any

# Relationship weights reflecting structural importance
RELATION_WEIGHTS: Dict[str, float] = {
    "CALLS": 1.0,
    "CONTAINS": 0.9,
    "INHERITS": 0.85,
    "IMPORTS": 0.8,
    "DEPENDS_ON": 0.75,
    "USES": 0.7,
    "DEFINES": 0.65,
}

DEFAULT_RELATION_WEIGHT = 0.6
DISTANCE_DECAY = 0.7  # Decay per hop distance beyond 1


def calculate_graph_score(relation_type: str, distance: int = 1) -> float:
    """Calculates a deterministic structural relevance score based on relationship type and hop distance."""
    rel_weight = RELATION_WEIGHTS.get(relation_type.upper(), DEFAULT_RELATION_WEIGHT)
    distance = max(1, int(distance))
    distance_factor = DISTANCE_DECAY ** (distance - 1)
    return round(rel_weight * distance_factor, 4)
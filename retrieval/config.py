
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for the hybrid retrieval pipeline."""

    dense_top_k: int = 20
    bm25_top_k: int = 20
    graph_top_k: int = 20
    rrf_top_k: int = 30
    rerank_top_k: int = 10
    rrf_k: int = 60
    fusion_weight: float | None = None

    def __post_init__(self) -> None:
        values = {
            "dense_top_k": self.dense_top_k,
            "bm25_top_k": self.bm25_top_k,
            "graph_top_k": self.graph_top_k,
            "rrf_top_k": self.rrf_top_k,
            "rerank_top_k": self.rerank_top_k,
            "rrf_k": self.rrf_k,
        }

        for name, value in values.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.fusion_weight is not None:
            if not 0.0 <= self.fusion_weight <= 1.0:
                raise ValueError("fusion_weight must be between 0 and 1")

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create a retrieval config with optional environment overrides."""
        defaults = cls()
        values = {}

        for name in (
            "dense_top_k",
            "bm25_top_k",
            "graph_top_k",
            "rrf_top_k",
            "rerank_top_k",
            "rrf_k",
        ):
            env_name = f"CODEGRAPHRAG_{name.upper()}"
            raw_value = os.getenv(env_name)

            if raw_value is None:
                values[name] = getattr(defaults, name)
                continue

            try:
                values[name] = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"{env_name} must be an integer, got {raw_value!r}"
                ) from exc

        raw_weight = os.getenv("CODEGRAPHRAG_FUSION_WEIGHT")
        if raw_weight is not None:
            try:
                values["fusion_weight"] = float(raw_weight)
            except ValueError as exc:
                raise ValueError(
                    "CODEGRAPHRAG_FUSION_WEIGHT must be a number"
                ) from exc

        return cls(**values)

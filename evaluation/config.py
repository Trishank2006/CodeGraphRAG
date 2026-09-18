from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration and metadata for a reproducible benchmark run."""

    dataset_version: str = "v1"
    top_k: int = 5
    rrf_k: int = 60
    rerank_top_k: int = 10

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "./models/ms-marco-MiniLM-L6-v2"
    llm_model: str = "gpt-5"

    max_context_items: int = 10
    max_context_characters: int = 12000

    repository_version: str = "unknown"
    benchmark_timestamp: str = ""

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than 0")

        if self.max_context_items <= 0:
            raise ValueError("max_context_items must be greater than 0")

        if self.max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than 0"
            )

    @classmethod
    def create(
        cls,
        *,
        repository_version: str = "unknown",
        **kwargs: Any,
    ) -> "EvaluationConfig":
        """Create a configuration with the current UTC timestamp."""

        return cls(
            repository_version=repository_version,
            benchmark_timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM_MODEL = "qwen2.5:3b"


def get_repository_version() -> str:
    """Return the current Git revision, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def get_git_working_tree_metadata() -> tuple[bool, str]:
    """Return dirty status and SHA-256 of tracked changes."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout

        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary"],
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout

        return (
            bool(status.strip()),
            hashlib.sha256(diff).hexdigest(),
        )
    except (OSError, subprocess.SubprocessError):
        return False, "unknown"


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration and metadata for a reproducible benchmark run."""

    dataset_version: str = "v1"
    top_k: int = 5
    rrf_k: int = 60
    rerank_top_k: int = 10

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    llm_model: str = DEFAULT_LLM_MODEL

    max_context_items: int = 10
    max_context_characters: int = 12000

    repository_version: str = "unknown"
    git_working_tree_dirty: bool = False
    tracked_diff_sha256: str = "unknown"
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
        repository_version: str | None = None,
        **kwargs: Any,
    ) -> "EvaluationConfig":
        """Create configuration with environment overrides and Git metadata."""
        dirty, diff_hash = get_git_working_tree_metadata()

        llm_model = (
            os.getenv("CODEGRAPHRAG_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_LLM_MODEL
        )

        reranker_model = os.getenv(
            "CODEGRAPHRAG_RERANKER_MODEL",
            DEFAULT_RERANKER_MODEL,
        )

        return cls(
            llm_model=llm_model,
            reranker_model=reranker_model,
            repository_version=(
                repository_version or get_repository_version()
            ),
            git_working_tree_dirty=dirty,
            tracked_diff_sha256=diff_hash,
            benchmark_timestamp=datetime.now(
                timezone.utc
            ).isoformat(),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)

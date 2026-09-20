from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv(override=True)


@dataclass(frozen=True)
class ApplicationSettings:
    """Runtime configuration read from environment variables."""

    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "code_chunks"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    openai_model: str = "llama-3.3-70b-versatile"

    @classmethod
    def from_environment(cls) -> "ApplicationSettings":
        model_name = (
            os.getenv("CODEGRAPHRAG_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "llama-3.3-70b-versatile"
        )
        return cls(
            qdrant_path=os.getenv("CODEGRAPHRAG_QDRANT_PATH", "data/qdrant"),
            qdrant_collection=os.getenv(
                "CODEGRAPHRAG_QDRANT_COLLECTION", "code_chunks"
            ),
            neo4j_uri=os.getenv("CODEGRAPHRAG_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("CODEGRAPHRAG_NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("CODEGRAPHRAG_NEO4J_PASSWORD", "password"),
            openai_model=model_name,
        )
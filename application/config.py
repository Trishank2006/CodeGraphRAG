from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from .env if present.
# Existing environment variables take precedence.
load_dotenv(override=False)


@dataclass(frozen=True)
class ApplicationSettings:
    """Runtime configuration read from environment variables."""

    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "code_chunks"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    openai_model: str = "llama-3.3-70b-versatile"

    @classmethod
    def from_environment(cls) -> "ApplicationSettings":
        """Create application settings from environment variables."""

        model_name = (
            os.getenv("CODEGRAPHRAG_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "llama-3.3-70b-versatile"
        )

        neo4j_password = os.getenv("CODEGRAPHRAG_NEO4J_PASSWORD")
        if not neo4j_password:
            raise RuntimeError(
                "CODEGRAPHRAG_NEO4J_PASSWORD must be set in your .env "
                "file or environment before starting CodeGraphRAG."
            )

        return cls(
            qdrant_path=os.getenv(
                "CODEGRAPHRAG_QDRANT_PATH",
                "data/qdrant",
            ),
            qdrant_collection=os.getenv(
                "CODEGRAPHRAG_QDRANT_COLLECTION",
                "code_chunks",
            ),
            neo4j_uri=os.getenv(
                "CODEGRAPHRAG_NEO4J_URI",
                "bolt://localhost:7687",
            ),
            neo4j_user=os.getenv(
                "CODEGRAPHRAG_NEO4J_USER",
                "neo4j",
            ),
            neo4j_password=neo4j_password,
            openai_model=model_name,
        )

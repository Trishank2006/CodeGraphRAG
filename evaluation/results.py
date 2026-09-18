from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class EvaluationResultsStore:
    """Persist benchmark results as JSON files."""

    def __init__(self, results_directory: str | Path = "evaluation/results"):
        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        name: str,
        results: Mapping[str, Any],
    ) -> Path:
        """Save one named evaluation result."""

        if not name.strip():
            raise ValueError("name must not be empty")

        filename = name.strip()

        if not filename.endswith(".json"):
            filename += ".json"

        path = self.results_directory / filename

        path.write_text(
            json.dumps(
                dict(results),
                indent=2,
            ),
            encoding="utf-8",
        )

        return path

    def load(
        self,
        name: str,
    ) -> dict[str, Any]:
        """Load one named evaluation result."""

        if not name.strip():
            raise ValueError("name must not be empty")

        filename = name.strip()

        if not filename.endswith(".json"):
            filename += ".json"

        path = self.results_directory / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation result not found: {path}"
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(data, dict):
            raise ValueError(
                "Evaluation result must contain a JSON object"
            )

        return data

    def exists(self, name: str) -> bool:
        """Return whether a named result exists."""

        filename = name.strip()

        if not filename.endswith(".json"):
            filename += ".json"

        return (self.results_directory / filename).exists()

    def list_results(self) -> list[str]:
        """Return stored result filenames in deterministic order."""

        return sorted(
            path.name
            for path in self.results_directory.glob("*.json")
        )

    def delete(self, name: str) -> None:
        """Delete one stored evaluation result."""

        filename = name.strip()

        if not filename.endswith(".json"):
            filename += ".json"

        path = self.results_directory / filename

        if path.exists():
            path.unlink()
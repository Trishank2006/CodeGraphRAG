from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class EntityMetadata:
    """Metadata representing a code entity in the knowledge graph."""
    id: str
    name: str
    label: str = "Node"
    file_path: str = ""
    language: str = "unknown"
    start_line: int = 0
    end_line: int = 0
    repository: str = "default_repo"


@dataclass(frozen=True)
class GraphEvidence:
    """Represents a discrete structural relationship between two repository code entities."""
    source_entity: str
    relationship: str
    target_entity: str
    distance: int = 1
    source_file: str = ""
    target_file: str = ""
    source_start_line: int = 0
    source_end_line: int = 0
    target_start_line: int = 0
    target_end_line: int = 0


@dataclass(frozen=True)
class PathStep:
    """Represents one directed transition in an execution/dependency path."""
    source: EntityMetadata
    relationship: str
    target: EntityMetadata


@dataclass(frozen=True)
class GraphPath:
    """Represents a complete traceable path through the codebase graph."""
    source: str
    target: str
    steps: List[PathStep] = field(default_factory=list)
    path_length: int = 0

    def format_path(self) -> str:
        """Serializes the path into a human- and LLM-readable execution chain."""
        if not self.steps:
            return f"{self.source} (no path found) {self.target}"
        
        parts = [self.steps[0].source.name]
        for step in self.steps:
            parts.append(f"--[{step.relationship}]--> {step.target.name}")
        return " ".join(parts)
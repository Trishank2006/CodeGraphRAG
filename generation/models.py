from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    """Structured source citation for generated answers."""

    file_path: str
    symbol: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ContextItem:
    """A single piece of repository evidence supplied to the LLM."""

    id: str
    file_path: str
    symbol: str
    start_line: int
    end_line: int
    content: str
    source: str


@dataclass(frozen=True)
class ContextPackage:
    """Bounded, structured context prepared for generation."""

    query: str
    items: tuple[ContextItem, ...] = field(default_factory=tuple)
    graph_context: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GeneratedAnswer:
    """Final grounded answer together with structured citations."""

    answer: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)
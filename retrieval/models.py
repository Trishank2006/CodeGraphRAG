from dataclasses import dataclass


@dataclass
class RetrievalResult:
    id: str
    source: str
    score: float
    repository: str
    file_path: str
    language: str
    symbol: str
    start_line: int
    end_line: int
    content: str
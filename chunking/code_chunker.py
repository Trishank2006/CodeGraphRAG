from dataclasses import dataclass


@dataclass
class CodeChunk:
    chunk_id: str
    repository: str
    file_path: str
    language: str
    symbol: str
    start_line: int
    end_line: int
    content: str


def create_chunk(
    repository: str,
    file_path: str,
    language: str,
    symbol: str,
    start_line: int,
    end_line: int,
    content: str,
) -> CodeChunk:
    """
    Create a structured code chunk for indexing.
    """
    chunk_id = (
        f"{repository}:{file_path}:"
        f"{start_line}:{end_line}"
    )

    return CodeChunk(
        chunk_id=chunk_id,
        repository=repository,
        file_path=file_path,
        language=language,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        content=content,
    )
from chunking.code_chunker import CodeChunk, create_chunk


def chunk_entities(
    repository: str,
    file_path: str,
    language: str,
    content: str,
    entities: list[dict],
) -> list[CodeChunk]:
    """
    Convert AST entities into code chunks.

    Each entity is expected to contain:
    - name
    - type
    - start_line
    - end_line
    """

    lines = content.splitlines()
    chunks = []

    for entity in entities:
        start_line = entity["start_line"]
        end_line = entity["end_line"]

        entity_content = "\n".join(
            lines[start_line - 1:end_line]
        )

        symbol = entity["name"]

        chunk = create_chunk(
            repository=repository,
            file_path=file_path,
            language=language,
            symbol=symbol,
            start_line=start_line,
            end_line=end_line,
            content=entity_content,
        )

        chunks.append(chunk)

    return chunks
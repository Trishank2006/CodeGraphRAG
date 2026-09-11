from chunking.code_chunker import CodeChunk, create_chunk


def chunk_entities(
    repository: str,
    file_path: str,
    language: str,
    content: str,
    entities: list,
) -> list[CodeChunk]:
    """
    Convert AST entities into retrieval-ready code chunks.

    Entities are expected to provide:
    - name
    - start_line
    - end_line

    Tree-sitter uses zero-based line numbers.
    Code chunks expose one-based line numbers.
    """

    lines = content.splitlines()
    chunks = []

    for entity in entities:
        start_line = entity.start_line
        end_line = entity.end_line

        # Tree-sitter line numbers are zero-based and end_line
        # refers to the last zero-based line of the entity.
        entity_content = "\n".join(
            lines[start_line:end_line + 1]
        )

        # Convert to one-based line numbers for our chunk metadata.
        chunk_start_line = start_line + 1
        chunk_end_line = end_line + 1

        chunk = create_chunk(
            repository=repository,
            file_path=file_path,
            language=language,
            symbol=entity.name,
            start_line=chunk_start_line,
            end_line=chunk_end_line,
            content=entity_content,
        )

        chunks.append(chunk)

    return chunks
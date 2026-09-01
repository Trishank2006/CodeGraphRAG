from pathlib import Path
import hashlib


def extract_metadata(
    file_path: str,
    repository: str = ""
) -> dict:
    """
    Extract metadata and source content from a file.

    Args:
        file_path: Path to the source file.
        repository: Repository name.

    Returns:
        Dictionary containing file metadata and content.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    content = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    file_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return {
        "repository": repository,
        "file_path": str(path),
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "size": path.stat().st_size,
        "line_count": len(content.splitlines()),
        "content": content,
        "file_hash": file_hash,
    }
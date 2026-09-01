from pathlib import Path

from ingestion.repository import clone_repository
from ingestion.file_discovery import discover_files
from ingestion.language_detection import detect_language
from ingestion.metadata import extract_metadata


def ingest_repository(
    repo_url: str,
    destination: str,
    repository_name: str | None = None
) -> list[dict]:
    """
    Clone and ingest a source-code repository.

    Args:
        repo_url: Git repository URL.
        destination: Local directory for the cloned repository.
        repository_name: Optional repository name.

    Returns:
        List of structured source-file dictionaries.
    """

    # Step 1: Clone repository
    repository_path = clone_repository(
        repo_url,
        destination
    )

    if repository_name is None:
        repository_name = Path(repo_url).stem

    # Step 2: Discover source files
    files = discover_files(
        str(repository_path)
    )

    results = []

    # Step 3: Process every source file
    for file_path in files:

        # Step 4: Detect programming language
        language = detect_language(
            str(file_path)
        )

        # Step 5: Extract metadata
        metadata = extract_metadata(
            str(file_path),
            repository=repository_name
        )

        metadata["language"] = language

        # Make path relative to repository
        metadata["file_path"] = str(
            Path(file_path).relative_to(repository_path)
        )

        results.append(metadata)

    return results
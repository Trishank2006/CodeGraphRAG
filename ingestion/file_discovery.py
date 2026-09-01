from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".hpp",
    ".go",
    ".rs",
}


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
}


def discover_files(repository_path: str) -> list[Path]:
    """
    Discover supported source-code files in a repository.

    Args:
        repository_path: Path to the local repository.

    Returns:
        List of source-code file paths.
    """

    root = Path(repository_path)

    if not root.exists():
        raise FileNotFoundError(
            f"Repository path does not exist: {repository_path}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Repository path is not a directory: {repository_path}"
        )

    files = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if any(
            directory in IGNORED_DIRECTORIES
            for directory in path.parts
        ):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return files
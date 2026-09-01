from pathlib import Path


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
}


def detect_language(file_path: str) -> str:
    """
    Detect the programming language of a source file
    based on its file extension.

    Args:
        file_path: Path to the source file.

    Returns:
        Detected programming language, or 'unknown'.
    """

    extension = Path(file_path).suffix.lower()

    return EXTENSION_TO_LANGUAGE.get(extension, "unknown")
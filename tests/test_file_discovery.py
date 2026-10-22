
from ingestion.file_discovery import discover_files


def test_discover_source_files(tmp_path):
    # Supported source files should be discovered.
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "app.js").write_text("console.log('hello')")

    # Unsupported files should be ignored.
    (tmp_path / "notes.txt").write_text("ignore me")

    # Existing ignored directory.
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "package.js").write_text("should be ignored")

    files = discover_files(str(tmp_path))
    file_names = {file.name for file in files}

    assert "main.py" in file_names
    assert "app.js" in file_names
    assert "notes.txt" not in file_names
    assert "package.js" not in file_names


def test_ignored_project_directories(tmp_path):
    # A normal project source file must remain discoverable.
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text("def main(): pass")

    # These directories must not enter the production index.
    ignored_dirs = ["evaluation", "tests", "temp_test_repo"]

    for dirname in ignored_dirs:
        directory = tmp_path / dirname
        directory.mkdir()
        (directory / "example.py").write_text("def example(): pass")

    files = discover_files(str(tmp_path))
    relative_paths = {
        file.relative_to(tmp_path).as_posix()
        for file in files
    }

    assert "src/main.py" in relative_paths

    for dirname in ignored_dirs:
        assert not any(
            path.startswith(f"{dirname}/")
            for path in relative_paths
        )

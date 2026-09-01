from ingestion.file_discovery import discover_files


def test_discover_source_files(tmp_path):

    # Create source files
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "app.js").write_text("console.log('hello')")

    # Create unsupported file
    (tmp_path / "notes.txt").write_text("ignore me")

    # Create directory that should be ignored
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()

    (node_modules / "package.js").write_text(
        "should be ignored"
    )

    files = discover_files(str(tmp_path))

    file_names = {file.name for file in files}

    assert "main.py" in file_names
    assert "app.js" in file_names
    assert "notes.txt" not in file_names
    assert "package.js" not in file_names
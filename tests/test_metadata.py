from ingestion.metadata import extract_metadata


def test_extract_metadata(tmp_path):

    source_file = tmp_path / "example.py"

    source_content = """def hello():
    return "Hello"

print(hello())
"""

    source_file.write_text(
        source_content,
        encoding="utf-8"
    )

    metadata = extract_metadata(
        str(source_file),
        repository="test-repository"
    )

    assert metadata["repository"] == "test-repository"
    assert metadata["file_name"] == "example.py"
    assert metadata["extension"] == ".py"
    assert metadata["content"] == source_content
    assert metadata["size"] == source_file.stat().st_size
    assert metadata["line_count"] == 4
    assert len(metadata["file_hash"]) == 64
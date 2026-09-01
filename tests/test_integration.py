from ingestion.main import ingest_repository


def test_ingestion_pipeline(tmp_path):

    # Use a small public repository for integration testing
    repo_url = "https://github.com/octocat/Hello-World.git"

    destination = tmp_path / "hello-world"

    results = ingest_repository(
        repo_url,
        str(destination),
        repository_name="hello-world"
    )

    assert isinstance(results, list)

    # Hello-World may contain no supported source files,
    # so we only verify that the pipeline executes correctly.
    for file_data in results:
        assert "repository" in file_data
        assert "file_path" in file_data
        assert "language" in file_data
        assert "content" in file_data
        assert "size" in file_data
        assert "file_hash" in file_data
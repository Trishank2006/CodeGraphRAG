from pathlib import Path

from ingestion.repository import clone_repository


def test_clone_repository(tmp_path):
    repo_url = "https://github.com/octocat/Hello-World.git"

    destination = tmp_path / "hello-world"

    result = clone_repository(
        repo_url,
        str(destination)
    )

    assert isinstance(result, Path)
    assert result.exists()
    assert (result / ".git").exists()
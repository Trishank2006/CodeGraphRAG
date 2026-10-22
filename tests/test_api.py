from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from application.service import IndexSummary


@dataclass
class FakeAnswer:
    answer: str
    citations: tuple = ()


class FakeService:
    def index_repository(self, repository_path, repository_name=None, exclude=()):
        # The API resolves paths before passing them on.
        assert Path(repository_path) == Path("/repo").resolve()
        return IndexSummary("demo", 1, 1, 1, 3, 2)

    def search(self, query, repository=None):
        assert query == "find run"
        return []

    def answer(self, query, repository=None):
        assert query == "explain run"
        return FakeAnswer("The run function executes the workflow.")


class RecordingService(FakeService):
    def __init__(self):
        self.searched_repositories = []

    def search(self, query, repository=None):
        self.searched_repositories.append(repository)
        return []


def test_api_exposes_index_search_and_answer_routes():
    client = TestClient(create_app(FakeService(), repository_root="/"))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/repositories/index", json={"repository_path": "/repo"}).json()[
        "graph_nodes"
    ] == 3
    assert client.post("/search", json={"query": "find run"}).json() == []
    assert client.post("/answers", json={"query": "explain run"}).json()["answer"] == (
        "The run function executes the workflow."
    )


def test_api_rejects_paths_outside_repository_root(tmp_path):
    client = TestClient(create_app(FakeService(), repository_root=tmp_path))

    response = client.post(
        "/repositories/index",
        json={"repository_path": str(tmp_path.parent)},
    )

    assert response.status_code == 403


def test_api_passes_repository_name_to_search():
    service = RecordingService()
    client = TestClient(create_app(service, repository_root="/"))

    client.post("/search", json={"query": "find run", "repository_name": "demo"})

    assert service.searched_repositories == ["demo"]
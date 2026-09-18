from dataclasses import dataclass

from fastapi.testclient import TestClient

from api.main import create_app
from application.service import IndexSummary


@dataclass
class FakeAnswer:
    answer: str
    citations: tuple = ()


class FakeService:
    def index_repository(self, repository_path, repository_name=None):
        assert repository_path == "/repo"
        return IndexSummary("demo", 1, 1, 1, 3, 2)

    def search(self, query):
        assert query == "find run"
        return []

    def answer(self, query):
        assert query == "explain run"
        return FakeAnswer("The run function executes the workflow.")


def test_api_exposes_index_search_and_answer_routes():
    client = TestClient(create_app(FakeService()))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/repositories/index", json={"repository_path": "/repo"}).json()[
        "graph_nodes"
    ] == 3
    assert client.post("/search", json={"query": "find run"}).json() == []
    assert client.post("/answers", json={"query": "explain run"}).json()["answer"] == (
        "The run function executes the workflow."
    )

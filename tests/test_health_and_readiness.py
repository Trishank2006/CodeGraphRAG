from fastapi.testclient import TestClient

from api.main import create_app
from graph.health import check_graph_health
from graph.neo4j_store import Neo4jStore


class FakeGraphStore:
    def __init__(self, database=None, fail=False):
        self.database = database
        self.fail = fail
        self.queries = []

    def execute_query(self, query, parameters=None):
        if self.fail:
            raise ConnectionError("Neo4j unavailable")
        self.queries.append(query)
        if "ping" in query:
            return [{"ping": 1}]
        if "node_count" in query:
            return [{"node_count": 5}]
        if "rel_count" in query:
            return [{"rel_count": 3}]
        return []


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def run(self, query, parameters=None, **kwargs):
        return []


class FakeDriver:
    def __init__(self):
        self.databases = []

    def session(self, database=None):
        self.databases.append(database)
        return FakeSession()

    def close(self):
        pass


class ReadyService:
    def __init__(self, checks):
        self.checks = checks

    def check_dependencies(self):
        return self.checks


def test_graph_health_reports_counts_for_a_healthy_store():
    result = check_graph_health(FakeGraphStore())

    assert result["status"] == "healthy"
    assert result["database"] == "neo4j"
    assert result["node_count"] == 5
    assert result["relationship_count"] == 3


def test_graph_health_reports_the_configured_database():
    result = check_graph_health(FakeGraphStore(database="analytics"))

    assert result["database"] == "analytics"


def test_graph_health_reports_failures_instead_of_raising():
    result = check_graph_health(FakeGraphStore(fail=True))

    assert result["status"] == "unhealthy"
    assert "Neo4j unavailable" in result["error"]


def test_graph_health_counts_without_a_cartesian_product():
    store = FakeGraphStore()

    check_graph_health(store)

    assert not any("OPTIONAL MATCH" in query for query in store.queries)


def test_neo4j_store_uses_the_configured_database():
    driver = FakeDriver()
    store = Neo4jStore(driver=driver, database="analytics")

    store.execute_query("RETURN 1")

    assert driver.databases == ["analytics"]


def test_neo4j_store_defaults_to_the_server_database(monkeypatch):
    monkeypatch.delenv("CODEGRAPHRAG_NEO4J_DATABASE", raising=False)
    driver = FakeDriver()
    store = Neo4jStore(driver=driver)

    store.execute_query("RETURN 1")

    assert driver.databases == [None]


def test_ready_returns_200_when_dependencies_are_healthy():
    service = ReadyService({
        "neo4j": {"status": "healthy"},
        "qdrant": {"status": "healthy"},
    })
    client = TestClient(create_app(service))

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_returns_503_when_a_dependency_is_down():
    service = ReadyService({
        "neo4j": {"status": "unhealthy", "error": "connection refused"},
        "qdrant": {"status": "healthy"},
    })
    client = TestClient(create_app(service))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["neo4j"]["status"] == "unhealthy"


def test_ready_returns_503_when_the_service_cannot_start(monkeypatch):
    import api.main as api_main

    def failing_service():
        raise RuntimeError("CODEGRAPHRAG_NEO4J_PASSWORD must be set")

    monkeypatch.setattr(api_main, "CodeGraphRAGService", failing_service)
    client = TestClient(api_main.create_app())

    response = client.get("/ready")

    assert response.status_code == 503

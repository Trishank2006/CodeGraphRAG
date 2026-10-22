
from application.service import CodeGraphRAGService


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.created = False
        self.chunks = []
        self.deleted_repositories = []

    def create_collection(self):
        self.created = True

    def delete_repository(self, repository):
        self.deleted_repositories.append(repository)

    def upsert_chunks(self, chunks, embeddings):
        self.chunks = chunks
        assert len(chunks) == len(embeddings)


class FakeGraphStore:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.constraints_created = False
        self.replaced_repositories = []

    def create_constraints(self):
        self.constraints_created = True

    def insert_nodes(self, nodes):
        self.nodes = nodes

    def insert_edges(self, edges):
        self.edges = edges

    def replace_repository(self, repository, nodes, edges):
        self.replaced_repositories.append(repository)
        self.nodes = nodes
        self.edges = edges

    def close(self):
        pass

def test_service_indexes_files_into_vector_and_graph_stores(tmp_path):
    source = tmp_path / "service.py"
    source.write_text(
        "import helper\n\ndef run():\n    execute()\n",
        encoding="utf-8",
    )

    vector_store = FakeVectorStore()
    graph_store = FakeGraphStore()

    service = CodeGraphRAGService(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        graph_store=graph_store,
    )

    summary = service.index_repository(str(tmp_path), "demo")

    assert summary.files == 1
    assert summary.chunks == 1
    assert vector_store.created is True
    assert vector_store.chunks[0]["symbol"] == "run"
    assert graph_store.constraints_created is True
    assert vector_store.deleted_repositories == ["demo"]
    assert graph_store.replaced_repositories == ["demo"]

    # Graph node and edge IDs are scoped to the indexed repository.
    assert any(
        node.id == "repo:demo:symbol:execute"
        for node in graph_store.nodes
    )
    assert any(
        edge.target_id == "repo:demo:module:helper"
        for edge in graph_store.edges
    )
def test_service_search_uses_retrieval_config_from_environment(
    monkeypatch,
):
    import application.service as service_module

    captured = {}

    class FakePipeline:
        def __init__(
            self,
            *,
            dense_search,
            bm25_retriever,
            graph_search,
            reranker,
            config,
        ):
            captured["config"] = config

        def search(self, query):
            return []

    class FakeGraphRetriever:
        def __init__(self, graph_store):
            pass

        def search(self, text, top_k):
            return []

    monkeypatch.setenv("CODEGRAPHRAG_RRF_TOP_K", "10")
    monkeypatch.setenv("CODEGRAPHRAG_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(
        service_module,
        "HybridRetrievalPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(
        service_module,
        "GraphRetriever",
        FakeGraphRetriever,
    )
    monkeypatch.setattr(
        service_module,
        "search_code",
        lambda **kwargs: [],
    )

    service = CodeGraphRAGService(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        graph_store=FakeGraphStore(),
    )

    service._require_index = lambda: None
    service.bm25_retriever = object()
    service.reranker = object()

    service.search("find authentication")

    assert captured["config"].rrf_top_k == 10


def test_service_preserves_existing_index_when_no_chunks_are_generated(
    tmp_path,
):
    source = tmp_path / "unsupported.xyz"
    source.write_text("not supported source code", encoding="utf-8")

    vector_store = FakeVectorStore()
    graph_store = FakeGraphStore()

    service = CodeGraphRAGService(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        graph_store=graph_store,
    )

    import pytest

    with pytest.raises(ValueError, match="No code chunks were generated"):
        service.index_repository(str(tmp_path), "demo")

    assert vector_store.deleted_repositories == []
    assert graph_store.replaced_repositories == []
    assert vector_store.created is False
    assert graph_store.constraints_created is False

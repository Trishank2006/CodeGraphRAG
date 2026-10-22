import pytest

from application.service import CodeGraphRAGService, _validate_before_write


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FailingEmbedder:
    def embed(self, texts):
        raise RuntimeError("model download failed")


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

    def replace_repository(self, repository, chunks, embeddings):
        self.delete_repository(repository)
        self.upsert_chunks(chunks=chunks, embeddings=embeddings)


class SizedFakeVectorStore(FakeVectorStore):
    vector_size = 384


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


class FailingGraphStore(FakeGraphStore):
    def replace_repository(self, repository, nodes, edges):
        raise ConnectionError("Neo4j unavailable")


def _write_source(tmp_path):
    (tmp_path / "service.py").write_text(
        "def run():\n    return 1\n",
        encoding="utf-8",
    )


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

    service._require_index = lambda repository=None: "demo"
    service._bm25_for = lambda repository: object()
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

    with pytest.raises(ValueError, match="No code chunks were generated"):
        service.index_repository(str(tmp_path), "demo")

    assert vector_store.deleted_repositories == []
    assert graph_store.replaced_repositories == []
    assert vector_store.created is False
    assert graph_store.constraints_created is False


# --- Re-index safety -----------------------------------------------------------

def test_failed_embedding_leaves_both_stores_untouched(tmp_path):
    _write_source(tmp_path)
    vector_store = FakeVectorStore()
    graph_store = FakeGraphStore()

    service = CodeGraphRAGService(
        embedder=FailingEmbedder(),
        vector_store=vector_store,
        graph_store=graph_store,
    )

    with pytest.raises(RuntimeError, match="model download failed"):
        service.index_repository(str(tmp_path), "demo")

    assert graph_store.replaced_repositories == []
    assert vector_store.deleted_repositories == []
    assert vector_store.chunks == []


def test_wrong_embedding_size_is_rejected_before_any_write(tmp_path):
    _write_source(tmp_path)
    vector_store = SizedFakeVectorStore()
    graph_store = FakeGraphStore()

    service = CodeGraphRAGService(
        embedder=FakeEmbedder(),  # 3-dimensional vectors; the store expects 384
        vector_store=vector_store,
        graph_store=graph_store,
    )

    with pytest.raises(ValueError, match="doesn't match the collection's vector size"):
        service.index_repository(str(tmp_path), "demo")

    assert graph_store.replaced_repositories == []
    assert vector_store.deleted_repositories == []


def test_graph_write_failure_leaves_vector_index_untouched(tmp_path):
    _write_source(tmp_path)
    vector_store = FakeVectorStore()

    service = CodeGraphRAGService(
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        graph_store=FailingGraphStore(),
    )

    with pytest.raises(ConnectionError, match="Neo4j unavailable"):
        service.index_repository(str(tmp_path), "demo")

    assert vector_store.deleted_repositories == []
    assert vector_store.chunks == []


def test_chunks_from_another_repository_are_rejected():
    with pytest.raises(ValueError, match="unexpected repositories"):
        _validate_before_write(
            "demo",
            [{"repository": "other"}],
            [[1.0, 0.0, 0.0]],
            3,
        )
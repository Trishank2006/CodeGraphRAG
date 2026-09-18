from application.service import CodeGraphRAGService


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.created = False
        self.chunks = []

    def create_collection(self):
        self.created = True

    def upsert_chunks(self, chunks, embeddings):
        self.chunks = chunks
        assert len(chunks) == len(embeddings)


class FakeGraphStore:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.constraints_created = False

    def create_constraints(self):
        self.constraints_created = True

    def insert_nodes(self, nodes):
        self.nodes = nodes

    def insert_edges(self, edges):
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
    assert any(node.id == "symbol:execute" for node in graph_store.nodes)
    assert any(edge.target_id == "module:helper" for edge in graph_store.edges)

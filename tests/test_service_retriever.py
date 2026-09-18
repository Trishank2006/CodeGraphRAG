from evaluation.service_retriever import ServiceRetriever


class FakeService:
    def __init__(self):
        self.embedder = object()
        self.vector_store = object()
        self.bm25_retriever = object()
        self.graph_store = object()
        self.reranker = object()


def test_service_retriever_initialization():
    service = FakeService()

    retriever = ServiceRetriever(service)

    assert retriever.service is service
    assert retriever.config is not None


def test_as_retrievers_contains_all_configurations():
    service = FakeService()

    retriever = ServiceRetriever(service)

    retrievers = retriever.as_retrievers()

    assert list(retrievers.keys()) == [
        "Dense",
        "BM25",
        "Dense + BM25",
        "Dense + BM25 + Graph",
        "Hybrid + RRF",
        "Hybrid + RRF + Reranker",
    ]


def test_retriever_functions_are_callable():
    service = FakeService()

    retriever = ServiceRetriever(service)

    retrievers = retriever.as_retrievers()

    for name, function in retrievers.items():
        assert callable(function)
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
import threading

from application.config import ApplicationSettings
from embeddings.embedder import CodeEmbedder
from generation.context import ContextBuilder
from generation.generator import AnswerGenerator
from generation.llm import LLMClient
from generation.models import GeneratedAnswer
from generation.openai_client import OpenAILLM
from generation.prompt import PromptBuilder
from graph.builder import GraphBuilder
from graph.grounding import GraphGroundingService
from graph.health import check_graph_health
from graph.neo4j_store import Neo4jStore
from graph.retrieval import GraphRetriever
from ingestion.file_discovery import discover_files
from ingestion.language_detection import detect_language
from ingestion.metadata import extract_metadata
from parser.core_parser import parse_source_code
from retrieval.bm25 import BM25Retriever
from retrieval.config import RetrievalConfig
from retrieval.models import RetrievalResult
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.reranker import CrossEncoderReranker
from retrieval.search import search_code
from vector_store.indexer import embed_chunks
from vector_store.qdrant_store import QdrantStore
from vector_store.repository_indexer import build_repository_chunks


VALID_RETRIEVAL_MODES = ("dense", "hybrid")


class RepositoryNotIndexedError(RuntimeError):
    """Raised when querying a repository that has no persisted index."""


class NoChunksGeneratedError(ValueError):
    """Raised when indexing produces no chunks; the existing index is untouched."""


@dataclass(frozen=True)
class IndexSummary:
    repository: str
    files: int
    entities: int
    chunks: int
    graph_nodes: int
    graph_edges: int


class CodeGraphRAGService:
    """Production composition root for indexing, retrieval, and answer generation."""

    def __init__(
        self,
        settings: ApplicationSettings | None = None,
        embedder: CodeEmbedder | None = None,
        vector_store: QdrantStore | None = None,
        graph_store: Neo4jStore | None = None,
        llm_client: LLMClient | None = None,
        bm25_retriever: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        retrieval_config: RetrievalConfig | None = None,
        retrieval_mode: str | None = None,
    ) -> None:
        self.settings = settings or ApplicationSettings.from_environment()
        self.embedder = embedder or CodeEmbedder()
        self.vector_store = vector_store or QdrantStore(
            path=self.settings.qdrant_path,
            collection_name=self.settings.qdrant_collection,
        )
        self.graph_store = graph_store or Neo4jStore(
            uri=self.settings.neo4j_uri,
            user=self.settings.neo4j_user,
            password=self.settings.neo4j_password,
        )
        self.llm_client = llm_client
        self.reranker = reranker

        mode = (
            retrieval_mode or os.getenv("CODEGRAPHRAG_RETRIEVAL_MODE", "dense")
        ).strip().lower()
        if mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(
                "CODEGRAPHRAG_RETRIEVAL_MODE must be 'dense' or 'hybrid'"
            )
        self.retrieval_mode = mode
        self.retrieval_config = retrieval_config or RetrievalConfig.from_env()

        # One lock guards repository state and Qdrant access, which every
        # API request shares.
        self._lock = threading.RLock()
        self._injected_bm25 = bm25_retriever
        self._bm25_by_repository: dict[str, BM25Retriever] = {}
        self._known_repositories: set[str] = set()
        self._is_indexed = False
        self._repository_name: str | None = None

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """BM25 index of the most recently indexed repository."""
        with self._lock:
            if self._repository_name in self._bm25_by_repository:
                return self._bm25_by_repository[self._repository_name]
            if self._injected_bm25 is None:
                self._injected_bm25 = BM25Retriever()
            return self._injected_bm25

    @bm25_retriever.setter
    def bm25_retriever(self, retriever: BM25Retriever) -> None:
        with self._lock:
            self._injected_bm25 = retriever
            if self._repository_name is not None:
                self._bm25_by_repository[self._repository_name] = retriever

    def check_dependencies(self) -> dict[str, dict]:
        """Report whether Neo4j and Qdrant are reachable."""
        graph = check_graph_health(self.graph_store)

        try:
            with self._lock:
                self.vector_store.ping()
            vectors = {"status": "healthy"}
        except Exception as exc:
            vectors = {"status": "unhealthy", "error": str(exc)}

        return {"neo4j": graph, "qdrant": vectors}

    def close(self) -> None:
        """Close all resources owned by the service."""
        try:
            self.graph_store.close()
        finally:
            self.vector_store.close()

    def index_repository(
        self,
        repository_path: str,
        repository_name: str | None = None,
        exclude: Iterable[str] = (),
    ) -> IndexSummary:
        """Parse, embed, and persist a checked-out source repository.

        Parsing, chunking, embedding, and validation all finish before
        anything is written, so a failure in those steps leaves the previous
        index intact.
        """
        root = Path(repository_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {root}")

        repository = repository_name or root.name
        excluded = _normalize_exclusions(exclude)
        files: list[dict] = []
        parsed_files = []
        entities_by_file: dict[str, list] = {}

        for source_path in discover_files(str(root)):
            relative_path = Path(source_path).relative_to(root).as_posix()
            if _is_excluded(relative_path, excluded):
                continue

            metadata = extract_metadata(str(source_path), repository=repository)
            metadata["file_path"] = relative_path
            metadata["language"] = detect_language(str(source_path))
            files.append(metadata)

            parsed = parse_source_code(
                file_path=relative_path,
                content=metadata["content"],
                language=metadata["language"],
            )
            parsed_files.append(parsed)
            entities_by_file[relative_path] = parsed.entities

        chunks = build_repository_chunks(files, entities_by_file)
        if not chunks:
            raise NoChunksGeneratedError(
                "No code chunks were generated; "
                "existing repository index was preserved."
            )

        graph_nodes, graph_edges = (
            GraphBuilder(repository).build_from_parsed_files(parsed_files)
        )

        # Embed before touching either store: this is the slowest step and
        # the one most likely to fail (model download, GPU memory, Ctrl+C).
        embeddings = embed_chunks(chunks, self.embedder)

        _validate_before_write(
            repository,
            chunks,
            embeddings,
            getattr(self.vector_store, "vector_size", None),
        )

        with self._lock:
            # Neo4j replaces the graph in one transaction and is the likelier
            # store to fail (network, auth), so it goes first; if it fails,
            # the vector index is still untouched.
            self.graph_store.create_constraints()
            self.graph_store.replace_repository(repository, graph_nodes, graph_edges)

            self.vector_store.create_collection()
            self.vector_store.replace_repository(repository, chunks, embeddings)

            bm25 = self._bm25_instance_for(repository)
            bm25.index(chunks)
            self._bm25_by_repository[repository] = bm25
            self._known_repositories.add(repository)
            self._is_indexed = True
            self._repository_name = repository

        return IndexSummary(
            repository=repository,
            files=len(files),
            entities=sum(len(parsed.entities) for parsed in parsed_files),
            chunks=len(chunks),
            graph_nodes=len(graph_nodes),
            graph_edges=len(graph_edges),
        )

    def has_repository(self, repository: str) -> bool:
        """Return True if the repository has a persisted index."""
        try:
            self._require_index(repository)
        except RepositoryNotIndexedError:
            return False
        return True

    def _require_index(self, repository: str | None = None) -> str:
        """Return the repository to query, checking it has a persisted index."""
        with self._lock:
            name = repository or self._repository_name
            if name is None:
                raise RepositoryNotIndexedError(
                    "Index a repository before searching or generating an answer"
                )

            if name not in self._known_repositories:
                if self.vector_store.count_repository(name) == 0:
                    raise RepositoryNotIndexedError(
                        f"Repository '{name}' has not been indexed"
                    )
                self._known_repositories.add(name)

            return name

    def _bm25_instance_for(self, repository: str) -> BM25Retriever:
        existing = self._bm25_by_repository.get(repository)
        if existing is not None:
            return existing

        injected = self._injected_bm25
        if injected is not None and all(
            retriever is not injected
            for retriever in self._bm25_by_repository.values()
        ):
            return injected

        return BM25Retriever()

    def _bm25_for(self, repository: str) -> BM25Retriever:
        """Return the repository's BM25 index, rebuilding it from Qdrant if needed."""
        with self._lock:
            retriever = self._bm25_by_repository.get(repository)
            if retriever is None:
                retriever = self._bm25_instance_for(repository)
                retriever.index(self.vector_store.load_repository_chunks(repository))
                self._bm25_by_repository[repository] = retriever
            return retriever

    def _dense_search(
        self, text: str, top_k: int, repository: str
    ) -> list[RetrievalResult]:
        with self._lock:
            return search_code(
                query=text,
                top_k=top_k,
                repository=repository,
                embedder=self.embedder,
                store=self.vector_store,
            )

    def search(
        self, query: str, repository: str | None = None
    ) -> list[RetrievalResult]:
        """Search one repository: dense by default, hybrid when configured."""
        repository_name = self._require_index(repository)
        config = self.retrieval_config
        dense_search = partial(self._dense_search, repository=repository_name)

        if self.retrieval_mode == "dense":
            return dense_search(query, config.dense_top_k)

        graph_retriever = GraphRetriever(self.graph_store)

        pipeline = HybridRetrievalPipeline(
            dense_search=dense_search,
            bm25_retriever=self._bm25_for(repository_name),
            graph_search=lambda text, top_k: graph_retriever.search(
                text,
                top_k=top_k,
                repository=repository_name,
            ),
            reranker=self.reranker,
            config=config,
        )

        return pipeline.search(query)

    def answer(
        self, query: str, repository: str | None = None
    ) -> GeneratedAnswer:
        """Generate an evidence-grounded answer for one indexed repository.

        In every retrieval mode, the context builder receives graph
        grounding from Neo4j.
        """
        repository_name = self._require_index(repository)
        grounding = GraphGroundingService(self.graph_store)

        generator = AnswerGenerator(
            retrieval_pipeline=_RepositoryRetriever(self, repository_name),
            context_builder=ContextBuilder(
                graph_context_getter=grounding.get_context,
            ),
            prompt_builder=PromptBuilder(),
            llm_client=self._get_llm_client(),
        )

        return generator.answer_query(query)

    def _get_llm_client(self) -> LLMClient:
        with self._lock:
            if self.llm_client is None:
                self.llm_client = OpenAILLM(model=self.settings.openai_model)
            return self.llm_client


class _RepositoryRetriever:
    """Binds AnswerGenerator's searches to one repository."""

    def __init__(self, service: CodeGraphRAGService, repository: str) -> None:
        self._service = service
        self._repository = repository

    def search(self, query: str) -> list[RetrievalResult]:
        return _drop_nested_results(self._service.search(query, self._repository))


def _drop_nested_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Drop results that contain, or sit inside, a higher-ranked result from
    the same file, so the LLM doesn't see the same code twice."""
    kept: list[RetrievalResult] = []
    for result in results:
        if not any(_nested(result, other) for other in kept):
            kept.append(result)
    return kept


def _nested(a: RetrievalResult, b: RetrievalResult) -> bool:
    if a.file_path != b.file_path:
        return False
    return (
        (a.start_line <= b.start_line and b.end_line <= a.end_line)
        or (b.start_line <= a.start_line and a.end_line <= b.end_line)
    )


def _normalize_exclusions(exclude: Iterable[str]) -> tuple[str, ...]:
    prefixes = []
    for item in exclude:
        prefix = item.strip().replace("\\", "/").removeprefix("./").strip("/")
        if prefix:
            prefixes.append(prefix)
    return tuple(prefixes)


def _is_excluded(relative_path: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        relative_path == prefix or relative_path.startswith(prefix + "/")
        for prefix in prefixes
    )


def _validate_before_write(
    repository: str,
    chunks: list[dict],
    embeddings,
    vector_size: int | None,
) -> None:
    """Catch problems that would otherwise surface halfway through the writes."""
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Got {len(embeddings)} embeddings for {len(chunks)} chunks"
        )

    unexpected = {chunk.get("repository") for chunk in chunks} - {repository}
    if unexpected:
        raise ValueError(
            "Chunks belong to unexpected repositories: "
            f"{sorted(map(str, unexpected))}"
        )

    if vector_size is not None:
        sizes = {len(vector) for vector in embeddings}
        if sizes != {vector_size}:
            raise ValueError(
                f"Embedding size {sorted(sizes)} doesn't match the collection's "
                f"vector size {vector_size}; the embedding model and the "
                "Qdrant collection are out of sync"
            )

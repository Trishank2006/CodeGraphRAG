from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embeddings.embedder import CodeEmbedder
from generation.context import ContextBuilder
from generation.generator import AnswerGenerator
from generation.llm import LLMClient
from generation.models import GeneratedAnswer
from generation.openai_client import OpenAILLM
from generation.prompt import PromptBuilder
from graph.builder import GraphBuilder
from graph.grounding import GraphGroundingService
from graph.neo4j_store import Neo4jStore
from graph.retrieval import GraphRetriever
from ingestion.file_discovery import discover_files
from ingestion.language_detection import detect_language
from ingestion.metadata import extract_metadata
from parser.core_parser import parse_source_code
from retrieval.bm25 import BM25Retriever
from retrieval.pipeline import HybridRetrievalPipeline
from retrieval.reranker import CrossEncoderReranker
from retrieval.search import search_code
from vector_store.indexer import index_chunks
from vector_store.qdrant_store import QdrantStore
from vector_store.repository_indexer import build_repository_chunks

from application.config import ApplicationSettings


class RepositoryNotIndexedError(RuntimeError):
    """Raised when querying a service before it has indexed repository data."""


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
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker or CrossEncoderReranker()
        self._is_indexed = False

    def close(self) -> None:
        self.graph_store.close()

    def index_repository(
        self,
        repository_path: str,
        repository_name: str | None = None,
    ) -> IndexSummary:
        """Parse and persist a checked-out source repository."""
        root = Path(repository_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {root}")

        repository = repository_name or root.name
        files: list[dict] = []
        parsed_files = []
        entities_by_file: dict[str, list] = {}

        for source_path in discover_files(str(root)):
            metadata = extract_metadata(str(source_path), repository=repository)
            relative_path = source_path.relative_to(root).as_posix()
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
        graph_nodes, graph_edges = GraphBuilder(repository).build_from_parsed_files(
            parsed_files
        )

        self.graph_store.create_constraints()
        self.graph_store.insert_nodes(graph_nodes)
        self.graph_store.insert_edges(graph_edges)
        index_chunks(chunks, embedder=self.embedder, store=self.vector_store)
        self.bm25_retriever.index(chunks)
        self._is_indexed = True

        return IndexSummary(
            repository=repository,
            files=len(files),
            entities=sum(len(parsed.entities) for parsed in parsed_files),
            chunks=len(chunks),
            graph_nodes=len(graph_nodes),
            graph_edges=len(graph_edges),
        )

    def _require_index(self) -> None:
        if not self._is_indexed:
            raise RepositoryNotIndexedError(
                "Index a repository before searching or generating an answer"
            )

    def search(self, query: str):
        """Run dense, BM25, and graph retrieval followed by reranking."""
        self._require_index()

        def dense_search(text: str, top_k: int):
            return search_code(
                query=text,
                top_k=top_k,
                embedder=self.embedder,
                store=self.vector_store,
            )

        graph_retriever = GraphRetriever(self.graph_store)
        pipeline = HybridRetrievalPipeline(
            dense_search=dense_search,
            bm25_retriever=self.bm25_retriever,
            graph_search=lambda text, top_k: graph_retriever.search(text, top_k=top_k),
            reranker=self.reranker,
        )
        return pipeline.search(query)

    def answer(self, query: str) -> GeneratedAnswer:
        """Generate an evidence-grounded answer using the configured LLM provider."""
        self._require_index()
        llm_client = self.llm_client or OpenAILLM(model=self.settings.openai_model)
        grounding = GraphGroundingService(self.graph_store)
        generator = AnswerGenerator(
            retrieval_pipeline=self,
            context_builder=ContextBuilder(
                graph_context_getter=grounding.get_context,
            ),
            prompt_builder=PromptBuilder(),
            llm_client=llm_client,
        )
        return generator.answer_query(query)

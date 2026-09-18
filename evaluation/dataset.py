from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class EvaluationExample:
    query: str
    query_type: str
    expected_files: tuple[str, ...] = field(default_factory=tuple)
    expected_symbols: tuple[str, ...] = field(default_factory=tuple)
    ground_truth_answer: str = ""


class EvaluationDataset:
    """Collection of manually verified evaluation examples."""

    def __init__(self, examples: Iterable[EvaluationExample] = ()) -> None:
        self._examples = tuple(examples)

    @property
    def examples(self) -> tuple[EvaluationExample, ...]:
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)

    def add(self, example: EvaluationExample) -> "EvaluationDataset":
        return EvaluationDataset((*self._examples, example))


def create_default_dataset() -> EvaluationDataset:
    """Return the initial manually verified functional benchmark dataset."""

    examples = [
        EvaluationExample(
            query="Where is the repository indexing pipeline implemented?",
            query_type="implementation_lookup",
            expected_files=("application/service.py",),
            expected_symbols=("CodeGraphRAGService.index_repository",),
            ground_truth_answer=(
                "The repository indexing pipeline is orchestrated by "
                "CodeGraphRAGService.index_repository in application/service.py."
            ),
        ),
        EvaluationExample(
            query="Where is the REST API application created?",
            query_type="implementation_lookup",
            expected_files=("api/main.py",),
            expected_symbols=("create_app",),
            ground_truth_answer=(
                "The REST API application is created by create_app in api/main.py."
            ),
        ),
        EvaluationExample(
            query="Where is the CLI entry point defined?",
            query_type="implementation_lookup",
            expected_files=("cli.py",),
            expected_symbols=("main",),
            ground_truth_answer=(
                "The CLI entry point is defined by main in cli.py."
            ),
        ),
        EvaluationExample(
            query="Where is hybrid retrieval configured?",
            query_type="implementation_lookup",
            expected_files=("retrieval/config.py",),
            expected_symbols=(),
            ground_truth_answer=(
                "Hybrid retrieval configuration is defined in retrieval/config.py."
            ),
        ),
        EvaluationExample(
            query="Where is BM25 retrieval implemented?",
            query_type="implementation_lookup",
            expected_files=("retrieval/bm25.py",),
            expected_symbols=("BM25Retriever",),
            ground_truth_answer=(
                "BM25 retrieval is implemented by BM25Retriever in retrieval/bm25.py."
            ),
        ),
        EvaluationExample(
            query="Where is reciprocal rank fusion implemented?",
            query_type="implementation_lookup",
            expected_files=("retrieval/rrf.py",),
            expected_symbols=("reciprocal_rank_fusion",),
            ground_truth_answer=(
                "Reciprocal rank fusion is implemented in retrieval/rrf.py."
            ),
        ),
        EvaluationExample(
            query="Where is the cross-encoder reranker implemented?",
            query_type="implementation_lookup",
            expected_files=("retrieval/reranker.py",),
            expected_symbols=("CrossEncoderReranker",),
            ground_truth_answer=(
                "The cross-encoder reranker is implemented by "
                "CrossEncoderReranker in retrieval/reranker.py."
            ),
        ),
        EvaluationExample(
            query="Where is the context builder implemented?",
            query_type="implementation_lookup",
            expected_files=("generation/context.py",),
            expected_symbols=("ContextBuilder",),
            ground_truth_answer=(
                "Context construction is implemented by ContextBuilder "
                "in generation/context.py."
            ),
        ),
        EvaluationExample(
            query="Where is the prompt builder implemented?",
            query_type="implementation_lookup",
            expected_files=("generation/prompt.py",),
            expected_symbols=("PromptBuilder",),
            ground_truth_answer=(
                "Prompt construction is implemented by PromptBuilder "
                "in generation/prompt.py."
            ),
        ),
        EvaluationExample(
            query="Where is the LLM abstraction defined?",
            query_type="symbol_lookup",
            expected_files=("generation/llm.py",),
            expected_symbols=("LLMClient",),
            ground_truth_answer=(
                "The LLM abstraction is defined by LLMClient in generation/llm.py."
            ),
        ),
        EvaluationExample(
            query="Where is the OpenAI LLM provider implemented?",
            query_type="implementation_lookup",
            expected_files=("generation/openai_client.py",),
            expected_symbols=("OpenAILLM",),
            ground_truth_answer=(
                "The OpenAI provider is implemented by OpenAILLM "
                "in generation/openai_client.py."
            ),
        ),
        EvaluationExample(
            query="Where is answer generation orchestrated?",
            query_type="implementation_lookup",
            expected_files=("generation/generator.py",),
            expected_symbols=("AnswerGenerator",),
            ground_truth_answer=(
                "Answer generation is orchestrated by AnswerGenerator "
                "in generation/generator.py."
            ),
        ),
        EvaluationExample(
            query="Where are generated-answer citations represented?",
            query_type="symbol_lookup",
            expected_files=("generation/models.py",),
            expected_symbols=("Citation",),
            ground_truth_answer=(
                "Generated-answer citations are represented by the Citation "
                "model in generation/models.py."
            ),
        ),
        EvaluationExample(
            query="Where is the repository indexer implemented?",
            query_type="implementation_lookup",
            expected_files=("vector_store/repository_indexer.py",),
            expected_symbols=("index_repository_files",),
            ground_truth_answer=(
                "Repository vector indexing is implemented by "
                "index_repository_files in vector_store/repository_indexer.py."
            ),
        ),
        EvaluationExample(
            query="Where is Qdrant persistence implemented?",
            query_type="implementation_lookup",
            expected_files=("vector_store/qdrant_store.py",),
            expected_symbols=("QdrantStore",),
            ground_truth_answer=(
                "Qdrant persistence is implemented by QdrantStore "
                "in vector_store/qdrant_store.py."
            ),
        ),
        EvaluationExample(
            query="Where is source-code parsing performed?",
            query_type="implementation_lookup",
            expected_files=("parser/core_parser.py",),
            expected_symbols=("parse_source_code",),
            ground_truth_answer=(
                "Source-code parsing is performed by parse_source_code "
                "in parser/core_parser.py."
            ),
        ),
        EvaluationExample(
            query="Where is repository cloning implemented?",
            query_type="implementation_lookup",
            expected_files=("ingestion/repository.py",),
            expected_symbols=("clone_repository",),
            ground_truth_answer=(
                "Repository cloning is implemented by clone_repository "
                "in ingestion/repository.py."
            ),
        ),
        EvaluationExample(
            query="Where is graph retrieval implemented?",
            query_type="implementation_lookup",
            expected_files=("graph/retrieval.py",),
            expected_symbols=("GraphRetriever",),
            ground_truth_answer=(
                "Graph retrieval is implemented by GraphRetriever "
                "in graph/retrieval.py."
            ),
        ),
        EvaluationExample(
            query="Where is graph grounding implemented?",
            query_type="implementation_lookup",
            expected_files=("graph/grounding.py",),
            expected_symbols=("GraphGroundingService",),
            ground_truth_answer=(
                "Graph grounding is implemented by GraphGroundingService "
                "in graph/grounding.py."
            ),
        ),
        EvaluationExample(
            query="Where is Neo4j persistence implemented?",
            query_type="implementation_lookup",
            expected_files=("graph/neo4j_store.py",),
            expected_symbols=("Neo4jStore",),
            ground_truth_answer=(
                "Neo4j persistence is implemented by Neo4jStore "
                "in graph/neo4j_store.py."
            ),
        ),
        EvaluationExample(
            query="Where are graph queries defined?",
            query_type="implementation_lookup",
            expected_files=("graph/queries.py",),
            expected_symbols=("GraphQueries",),
            ground_truth_answer=(
                "Graph query operations are defined by GraphQueries "
                "in graph/queries.py."
            ),
        ),
        EvaluationExample(
            query="Where are graph health checks implemented?",
            query_type="implementation_lookup",
            expected_files=("graph/health.py",),
            expected_symbols=(),
            ground_truth_answer=(
                "Graph health checks are implemented in graph/health.py."
            ),
        ),
        EvaluationExample(
            query="Where is repository-level graph management implemented?",
            query_type="implementation_lookup",
            expected_files=("graph/repository.py",),
            expected_symbols=(),
            ground_truth_answer=(
                "Repository-level graph management is implemented "
                "in graph/repository.py."
            ),
        ),
        EvaluationExample(
            query="What component combines dense and lexical retrieval?",
            query_type="conceptual",
            expected_files=("retrieval/hybrid.py",),
            expected_symbols=("HybridRetriever",),
            ground_truth_answer=(
                "The hybrid retrieval component combines dense retrieval "
                "with lexical BM25 retrieval."
            ),
        ),
        EvaluationExample(
            query="What is the purpose of RRF in the retrieval pipeline?",
            query_type="conceptual",
            expected_files=("retrieval/rrf.py", "retrieval/hybrid.py"),
            expected_symbols=(),
            ground_truth_answer=(
                "RRF combines ranked results from multiple retrieval sources "
                "into a unified ranking."
            ),
        ),
        EvaluationExample(
            query="What does the context builder control?",
            query_type="conceptual",
            expected_files=("generation/context.py",),
            expected_symbols=("ContextBuilder",),
            ground_truth_answer=(
                "The context builder selects and limits retrieved evidence "
                "before it is supplied to the generation stage."
            ),
        ),
        EvaluationExample(
            query="What does the application service orchestrate?",
            query_type="conceptual",
            expected_files=("application/service.py",),
            expected_symbols=("CodeGraphRAGService",),
            ground_truth_answer=(
                "The application service orchestrates repository indexing, "
                "retrieval, grounding, and answer generation."
            ),
        ),
        EvaluationExample(
            query="Which component provides grounded answers?",
            query_type="functional",
            expected_files=("generation/generator.py", "application/service.py"),
            expected_symbols=("AnswerGenerator", "CodeGraphRAGService"),
            ground_truth_answer=(
                "Grounded answers are produced through the application service "
                "and AnswerGenerator."
            ),
        ),
        EvaluationExample(
            query="How are code search results represented?",
            query_type="conceptual",
            expected_files=("retrieval/models.py",),
            expected_symbols=("RetrievalResult",),
            ground_truth_answer=(
                "Code search results are represented using the RetrievalResult "
                "data model."
            ),
        ),
        EvaluationExample(
            query="Which component stores code embeddings?",
            query_type="functional",
            expected_files=("vector_store/qdrant_store.py",),
            expected_symbols=("QdrantStore",),
            ground_truth_answer=(
                "Code embeddings are stored through the QdrantStore component."
            ),
        ),
        EvaluationExample(
            query="How does the system expose repository indexing?",
            query_type="functional",
            expected_files=("api/main.py", "application/service.py"),
            expected_symbols=("create_app", "CodeGraphRAGService.index_repository"),
            ground_truth_answer=(
                "Repository indexing is exposed through the REST API and "
                "delegated to CodeGraphRAGService.index_repository."
            ),
        ),
    ]

    return EvaluationDataset(examples)
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
        EvaluationExample(
            query='Where is AST-aware chunk extraction implemented?',
            query_type='implementation_lookup',
            expected_files=('chunking/ast_chunker.py',),
            expected_symbols=('chunk_entities',),
            ground_truth_answer='AST-aware chunk extraction is implemented by chunk_entities in chunking/ast_chunker.py.',
        ),
        EvaluationExample(
            query='Where is the CodeChunk data model defined?',
            query_type='symbol_lookup',
            expected_files=('chunking/code_chunker.py',),
            expected_symbols=('CodeChunk',),
            ground_truth_answer='The CodeChunk data model is defined in chunking/code_chunker.py.',
        ),
        EvaluationExample(
            query='Where are code chunks created?',
            query_type='implementation_lookup',
            expected_files=('chunking/code_chunker.py',),
            expected_symbols=('create_chunk',),
            ground_truth_answer='Code chunks are created by create_chunk in chunking/code_chunker.py.',
        ),
        EvaluationExample(
            query='Where is the code embedder defined?',
            query_type='symbol_lookup',
            expected_files=('embeddings/embedder.py',),
            expected_symbols=('CodeEmbedder',),
            ground_truth_answer='The CodeEmbedder class is defined in embeddings/embedder.py.',
        ),
        EvaluationExample(
            query='Where are text embeddings generated?',
            query_type='implementation_lookup',
            expected_files=('embeddings/embedder.py',),
            expected_symbols=('CodeEmbedder.embed',),
            ground_truth_answer='Text embeddings are generated by CodeEmbedder.embed in embeddings/embedder.py.',
        ),
        EvaluationExample(
            query='Where are query embeddings generated?',
            query_type='implementation_lookup',
            expected_files=('embeddings/embedder.py',),
            expected_symbols=('CodeEmbedder.embed_query',),
            ground_truth_answer='Query embeddings are generated by CodeEmbedder.embed_query in embeddings/embedder.py.',
        ),
        EvaluationExample(
            query='Where is the application configuration model defined?',
            query_type='symbol_lookup',
            expected_files=('application/config.py',),
            expected_symbols=('ApplicationSettings',),
            ground_truth_answer='Application configuration is represented by ApplicationSettings in application/config.py.',
        ),
        EvaluationExample(
            query='Where are environment variables loaded into application settings?',
            query_type='implementation_lookup',
            expected_files=('application/config.py',),
            expected_symbols=('from_environment',),
            ground_truth_answer='Environment-based settings are loaded by from_environment in application/config.py.',
        ),
        EvaluationExample(
            query='Where is the error for querying an unindexed repository defined?',
            query_type='symbol_lookup',
            expected_files=('application/service.py',),
            expected_symbols=('RepositoryNotIndexedError',),
            ground_truth_answer='RepositoryNotIndexedError is defined in application/service.py.',
        ),
        EvaluationExample(
            query='Where is the repository indexing summary model defined?',
            query_type='symbol_lookup',
            expected_files=('application/service.py',),
            expected_symbols=('IndexSummary',),
            ground_truth_answer='The indexing summary model IndexSummary is defined in application/service.py.',
        ),
        EvaluationExample(
            query='Where does the service check whether an index is available?',
            query_type='implementation_lookup',
            expected_files=('application/service.py',),
            expected_symbols=('CodeGraphRAGService._require_index',),
            ground_truth_answer='The service checks index availability in CodeGraphRAGService._require_index in application/service.py.',
        ),
        EvaluationExample(
            query='Where is the service search operation implemented?',
            query_type='implementation_lookup',
            expected_files=('application/service.py',),
            expected_symbols=('CodeGraphRAGService.search',),
            ground_truth_answer='The service search operation is implemented by CodeGraphRAGService.search in application/service.py.',
        ),
        EvaluationExample(
            query='Where is the service answer operation implemented?',
            query_type='implementation_lookup',
            expected_files=('application/service.py',),
            expected_symbols=('CodeGraphRAGService.answer',),
            ground_truth_answer='The service answer operation is implemented by CodeGraphRAGService.answer in application/service.py.',
        ),
        EvaluationExample(
            query='Where is the API request model for indexing defined?',
            query_type='symbol_lookup',
            expected_files=('api/main.py',),
            expected_symbols=('IndexRequest',),
            ground_truth_answer='The API indexing request model is defined as IndexRequest in api/main.py.',
        ),
        EvaluationExample(
            query='Where is the API request model for queries defined?',
            query_type='symbol_lookup',
            expected_files=('api/main.py',),
            expected_symbols=('QueryRequest',),
            ground_truth_answer='The API query request model is defined as QueryRequest in api/main.py.',
        ),
        EvaluationExample(
            query='Where is the API health endpoint implemented?',
            query_type='implementation_lookup',
            expected_files=('api/main.py',),
            expected_symbols=('health',),
            ground_truth_answer='The API health endpoint is implemented by health in api/main.py.',
        ),
        EvaluationExample(
            query='Where is the API repository-indexing endpoint implemented?',
            query_type='implementation_lookup',
            expected_files=('api/main.py',),
            expected_symbols=('index_repository',),
            ground_truth_answer='The API repository-indexing endpoint is implemented in api/main.py.',
        ),
        EvaluationExample(
            query='Where is the API search endpoint implemented?',
            query_type='implementation_lookup',
            expected_files=('api/main.py',),
            expected_symbols=('search',),
            ground_truth_answer='The API search endpoint is implemented in api/main.py.',
        ),
        EvaluationExample(
            query='Where is the API answer endpoint implemented?',
            query_type='implementation_lookup',
            expected_files=('api/main.py',),
            expected_symbols=('answer',),
            ground_truth_answer='The API answer endpoint is implemented in api/main.py.',
        ),
        EvaluationExample(
            query='Where is CLI argument parsing configured?',
            query_type='implementation_lookup',
            expected_files=('cli.py',),
            expected_symbols=('build_parser',),
            ground_truth_answer='CLI argument parsing is configured by build_parser in cli.py.',
        ),
        EvaluationExample(
            query='Where is CLI output for indexing summaries formatted?',
            query_type='implementation_lookup',
            expected_files=('cli.py',),
            expected_symbols=('_print_index_summary',),
            ground_truth_answer='CLI indexing-summary output is formatted by _print_index_summary in cli.py.',
        ),
        EvaluationExample(
            query='Where is CLI output for answers formatted?',
            query_type='implementation_lookup',
            expected_files=('cli.py',),
            expected_symbols=('_print_answer',),
            ground_truth_answer='CLI answer output is formatted by _print_answer in cli.py.',
        ),
        EvaluationExample(
            query='Where is JSON output formatting for the CLI implemented?',
            query_type='implementation_lookup',
            expected_files=('cli.py',),
            expected_symbols=('_print_json',),
            ground_truth_answer='CLI JSON output formatting is implemented by _print_json in cli.py.',
        ),
        EvaluationExample(
            query='Where is the repository ingestion entry point defined?',
            query_type='implementation_lookup',
            expected_files=('ingestion/main.py',),
            expected_symbols=('ingest_repository',),
            ground_truth_answer='The repository ingestion entry point is defined by ingest_repository in ingestion/main.py.',
        ),
        EvaluationExample(
            query='Where are repository files discovered?',
            query_type='implementation_lookup',
            expected_files=('ingestion/file_discovery.py',),
            expected_symbols=('discover_files',),
            ground_truth_answer='Repository file discovery is implemented by discover_files in ingestion/file_discovery.py.',
        ),
        EvaluationExample(
            query='Where is programming-language detection implemented?',
            query_type='implementation_lookup',
            expected_files=('ingestion/language_detection.py',),
            expected_symbols=('detect_language',),
            ground_truth_answer='Programming-language detection is implemented by detect_language in ingestion/language_detection.py.',
        ),
        EvaluationExample(
            query='Where is source-file metadata extracted?',
            query_type='implementation_lookup',
            expected_files=('ingestion/metadata.py',),
            expected_symbols=('extract_metadata',),
            ground_truth_answer='Source-file metadata extraction is implemented by extract_metadata in ingestion/metadata.py.',
        ),
        EvaluationExample(
            query='Where is repository ingestion initialized?',
            query_type='implementation_lookup',
            expected_files=('ingestion/main.py',),
            expected_symbols=('ingest_repository',),
            ground_truth_answer='Repository ingestion is initialized through ingest_repository in ingestion/main.py.',
        ),
        EvaluationExample(
            query='Where is the parsed code entity model defined?',
            query_type='symbol_lookup',
            expected_files=('parser/models.py',),
            expected_symbols=('CodeEntity',),
            ground_truth_answer='The parsed code entity model is defined by CodeEntity in parser/models.py.',
        ),
        EvaluationExample(
            query='Where is the parsed code relationship model defined?',
            query_type='symbol_lookup',
            expected_files=('parser/models.py',),
            expected_symbols=('CodeRelationship',),
            ground_truth_answer='The parsed code relationship model is defined by CodeRelationship in parser/models.py.',
        ),
        EvaluationExample(
            query='Where is the parsed-file model defined?',
            query_type='symbol_lookup',
            expected_files=('parser/models.py',),
            expected_symbols=('ParsedFile',),
            ground_truth_answer='The parsed-file model is defined by ParsedFile in parser/models.py.',
        ),
        EvaluationExample(
            query='Where is graph construction from parsed files implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/builder.py',),
            expected_symbols=('GraphBuilder.build_from_parsed_files',),
            ground_truth_answer='Graph construction from parsed files is implemented by GraphBuilder.build_from_parsed_files in graph/builder.py.',
        ),
        EvaluationExample(
            query='Where are scoped graph identifiers generated?',
            query_type='implementation_lookup',
            expected_files=('graph/builder.py',),
            expected_symbols=('GraphBuilder._scoped_id',),
            ground_truth_answer='Scoped graph identifiers are generated by GraphBuilder._scoped_id in graph/builder.py.',
        ),
        EvaluationExample(
            query='Where are reference nodes added to the graph?',
            query_type='implementation_lookup',
            expected_files=('graph/builder.py',),
            expected_symbols=('GraphBuilder.add_reference_node',),
            ground_truth_answer='Reference nodes are added by GraphBuilder.add_reference_node in graph/builder.py.',
        ),
        EvaluationExample(
            query='Where is graph node metadata represented?',
            query_type='symbol_lookup',
            expected_files=('graph/evidence.py',),
            expected_symbols=('EntityMetadata',),
            ground_truth_answer='Graph node metadata is represented by EntityMetadata in graph/evidence.py.',
        ),
        EvaluationExample(
            query='Where is graph evidence represented?',
            query_type='symbol_lookup',
            expected_files=('graph/evidence.py',),
            expected_symbols=('GraphEvidence',),
            ground_truth_answer='Graph evidence is represented by GraphEvidence in graph/evidence.py.',
        ),
        EvaluationExample(
            query='Where are graph path steps represented?',
            query_type='symbol_lookup',
            expected_files=('graph/evidence.py',),
            expected_symbols=('PathStep',),
            ground_truth_answer='Graph path steps are represented by PathStep in graph/evidence.py.',
        ),
        EvaluationExample(
            query='Where is a graph path represented?',
            query_type='symbol_lookup',
            expected_files=('graph/evidence.py',),
            expected_symbols=('GraphPath',),
            ground_truth_answer='A graph path is represented by GraphPath in graph/evidence.py.',
        ),
        EvaluationExample(
            query='Where is graph path formatting implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/evidence.py',),
            expected_symbols=('format_path',),
            ground_truth_answer='Graph path formatting is implemented by format_path in graph/evidence.py.',
        ),
        EvaluationExample(
            query='Where is graph context extraction implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/context.py',),
            expected_symbols=('GraphContextExtractor',),
            ground_truth_answer='Graph context extraction is implemented by GraphContextExtractor in graph/context.py.',
        ),
        EvaluationExample(
            query='Where is graph hierarchy formatting implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/context.py',),
            expected_symbols=('format_hierarchy',),
            ground_truth_answer='Graph hierarchy formatting is implemented by format_hierarchy in graph/context.py.',
        ),
        EvaluationExample(
            query='Where is graph evidence retrieval implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/grounding.py',),
            expected_symbols=('GraphGroundingService.get_evidence',),
            ground_truth_answer='Graph evidence retrieval is implemented by GraphGroundingService.get_evidence in graph/grounding.py.',
        ),
        EvaluationExample(
            query='Where is graph grounding context retrieval implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/grounding.py',),
            expected_symbols=('GraphGroundingService.get_context',),
            ground_truth_answer='Graph grounding context retrieval is implemented by GraphGroundingService.get_context in graph/grounding.py.',
        ),
        EvaluationExample(
            query='Where is graph path lookup implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/grounding.py',),
            expected_symbols=('GraphGroundingService.get_path',),
            ground_truth_answer='Graph path lookup is implemented by GraphGroundingService.get_path in graph/grounding.py.',
        ),
        EvaluationExample(
            query='Where are graph node types defined?',
            query_type='symbol_lookup',
            expected_files=('graph/schema.py',),
            expected_symbols=('NodeType',),
            ground_truth_answer='Graph node types are defined by NodeType in graph/schema.py.',
        ),
        EvaluationExample(
            query='Where are graph relationship types defined?',
            query_type='symbol_lookup',
            expected_files=('graph/schema.py',),
            expected_symbols=('RelationType',),
            ground_truth_answer='Graph relationship types are defined by RelationType in graph/schema.py.',
        ),
        EvaluationExample(
            query='Where is graph relationship path finding implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/path.py',),
            expected_symbols=('GraphPathFinder.find_relationship',),
            ground_truth_answer='Graph relationship path finding is implemented by GraphPathFinder.find_relationship in graph/path.py.',
        ),
        EvaluationExample(
            query='Where is graph path retrieval implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/path.py',),
            expected_symbols=('GraphPathFinder.get_path',),
            ground_truth_answer='Graph path retrieval is implemented by GraphPathFinder.get_path in graph/path.py.',
        ),
        EvaluationExample(
            query='Where are graph repository statistics retrieved?',
            query_type='implementation_lookup',
            expected_files=('graph/repository.py',),
            expected_symbols=('GraphRepositoryManager.get_repository_stats',),
            ground_truth_answer='Graph repository statistics are retrieved by GraphRepositoryManager.get_repository_stats in graph/repository.py.',
        ),
        EvaluationExample(
            query='Where is graph repository deletion implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/repository.py',),
            expected_symbols=('GraphRepositoryManager.delete_repository',),
            ground_truth_answer='Graph repository deletion is implemented by GraphRepositoryManager.delete_repository in graph/repository.py.',
        ),
        EvaluationExample(
            query='Where are function callers queried from the graph?',
            query_type='implementation_lookup',
            expected_files=('graph/queries.py',),
            expected_symbols=('GraphQueries.get_function_callers',),
            ground_truth_answer='Function-caller queries are implemented by GraphQueries.get_function_callers in graph/queries.py.',
        ),
        EvaluationExample(
            query='Where are function callees queried from the graph?',
            query_type='implementation_lookup',
            expected_files=('graph/queries.py',),
            expected_symbols=('GraphQueries.get_function_callees',),
            ground_truth_answer='Function-callee queries are implemented by GraphQueries.get_function_callees in graph/queries.py.',
        ),
        EvaluationExample(
            query='Where are file dependencies queried from the graph?',
            query_type='implementation_lookup',
            expected_files=('graph/queries.py',),
            expected_symbols=('GraphQueries.get_file_dependencies',),
            ground_truth_answer='File-dependency queries are implemented by GraphQueries.get_file_dependencies in graph/queries.py.',
        ),
        EvaluationExample(
            query='Where are class methods queried from the graph?',
            query_type='implementation_lookup',
            expected_files=('graph/queries.py',),
            expected_symbols=('GraphQueries.get_class_methods',),
            ground_truth_answer='Class-method queries are implemented by GraphQueries.get_class_methods in graph/queries.py.',
        ),
        EvaluationExample(
            query='Where are parent classes queried from the graph?',
            query_type='implementation_lookup',
            expected_files=('graph/queries.py',),
            expected_symbols=('GraphQueries.get_parent_classes',),
            ground_truth_answer='Parent-class queries are implemented by GraphQueries.get_parent_classes in graph/queries.py.',
        ),
        EvaluationExample(
            query='Where are file symbols queried from the graph?',
            query_type='implementation_lookup',
            expected_files=('graph/queries.py',),
            expected_symbols=('GraphQueries.get_file_symbols',),
            ground_truth_answer='File-symbol queries are implemented by GraphQueries.get_file_symbols in graph/queries.py.',
        ),
        EvaluationExample(
            query='Where is graph neighbor expansion implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/queries.py',),
            expected_symbols=('GraphQueries.get_neighbors_with_depth',),
            ground_truth_answer='Graph neighbor expansion is implemented by GraphQueries.get_neighbors_with_depth in graph/queries.py.',
        ),
        EvaluationExample(
            query='Where is graph-based ranking implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/ranking.py',),
            expected_symbols=('calculate_graph_score',),
            ground_truth_answer='Graph-based ranking is implemented by calculate_graph_score in graph/ranking.py.',
        ),
        EvaluationExample(
            query='Where is graph call-chain traversal implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/traversal.py',),
            expected_symbols=('GraphTraversal.get_call_chain',),
            ground_truth_answer='Graph call-chain traversal is implemented by GraphTraversal.get_call_chain in graph/traversal.py.',
        ),
        EvaluationExample(
            query='Where is the graph node data model defined?',
            query_type='symbol_lookup',
            expected_files=('graph/models.py',),
            expected_symbols=('GraphNode',),
            ground_truth_answer='The graph node data model is defined by GraphNode in graph/models.py.',
        ),
        EvaluationExample(
            query='Where is the graph edge data model defined?',
            query_type='symbol_lookup',
            expected_files=('graph/models.py',),
            expected_symbols=('GraphEdge',),
            ground_truth_answer='The graph edge data model is defined by GraphEdge in graph/models.py.',
        ),
        EvaluationExample(
            query='Where is the Neo4j repository replacement operation implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/neo4j_store.py',),
            expected_symbols=('Neo4jStore.replace_repository',),
            ground_truth_answer='Neo4j repository replacement is implemented by Neo4jStore.replace_repository in graph/neo4j_store.py.',
        ),
        EvaluationExample(
            query='Where are graph nodes inserted into Neo4j?',
            query_type='implementation_lookup',
            expected_files=('graph/neo4j_store.py',),
            expected_symbols=('Neo4jStore.insert_nodes',),
            ground_truth_answer='Graph node insertion is implemented by Neo4jStore.insert_nodes in graph/neo4j_store.py.',
        ),
        EvaluationExample(
            query='Where are graph edges inserted into Neo4j?',
            query_type='implementation_lookup',
            expected_files=('graph/neo4j_store.py',),
            expected_symbols=('Neo4jStore.insert_edges',),
            ground_truth_answer='Graph edge insertion is implemented by Neo4jStore.insert_edges in graph/neo4j_store.py.',
        ),
        EvaluationExample(
            query='Where is graph health checking implemented?',
            query_type='implementation_lookup',
            expected_files=('graph/health.py',),
            expected_symbols=('check_graph_health',),
            ground_truth_answer='Graph health checking is implemented by check_graph_health in graph/health.py.',
        ),
        EvaluationExample(
            query='Where is the retrieval result data model defined?',
            query_type='symbol_lookup',
            expected_files=('retrieval/models.py',),
            expected_symbols=('RetrievalResult',),
            ground_truth_answer='The retrieval result data model is defined by RetrievalResult in retrieval/models.py.',
        ),
        EvaluationExample(
            query='Where is weighted rank fusion implemented?',
            query_type='implementation_lookup',
            expected_files=('retrieval/pipeline.py',),
            expected_symbols=('weighted_rank_fusion',),
            ground_truth_answer='Weighted rank fusion is implemented by weighted_rank_fusion in retrieval/pipeline.py.',
        ),
        EvaluationExample(
            query='Where is the hybrid retrieval pipeline defined?',
            query_type='symbol_lookup',
            expected_files=('retrieval/pipeline.py',),
            expected_symbols=('HybridRetrievalPipeline',),
            ground_truth_answer='The hybrid retrieval pipeline is defined by HybridRetrievalPipeline in retrieval/pipeline.py.',
        ),
        EvaluationExample(
            query='Where is code search exposed as a helper function?',
            query_type='implementation_lookup',
            expected_files=('retrieval/search.py',),
            expected_symbols=('search_code',),
            ground_truth_answer='The code-search helper is implemented by search_code in retrieval/search.py.',
        ),
    ]

    return EvaluationDataset(examples)
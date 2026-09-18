# CodeGraphRAG architecture

CodeGraphRAG indexes a checked-out source repository into two complementary
stores: Qdrant for semantic vector retrieval and Neo4j for structural code
relationships.

```text
local repository
  -> discovery + metadata
  -> Tree-sitter entities/relationships
  -> AST-aware chunks -> BGE embeddings -> Qdrant
  -> graph nodes/edges -> Neo4j

query
  -> dense + BM25 + graph retrieval
  -> reciprocal-rank fusion + cross-encoder reranking
  -> bounded code and graph context
  -> LLM answer with structured citations
```

## Production service

`application.service.CodeGraphRAGService` is the composition root. Its
`index_repository()` method writes both stores and builds the in-memory BM25
index for the active application process. `search()` executes the hybrid
retrieval pipeline and `answer()` uses `GraphGroundingService` plus the
configured LLM provider.

Neo4j nodes always receive the `:Node` label used by graph queries. Imports
and unresolved call targets are persisted as reference nodes whose `label`
property is `Module` or `Symbol`, so their relationships are not dropped
during insertion.

## Configuration

The application reads these optional environment variables:

- `CODEGRAPHRAG_QDRANT_PATH` and `CODEGRAPHRAG_QDRANT_COLLECTION`
- `CODEGRAPHRAG_NEO4J_URI`, `CODEGRAPHRAG_NEO4J_USER`, and
  `CODEGRAPHRAG_NEO4J_PASSWORD`
- `CODEGRAPHRAG_OPENAI_MODEL`
- `OPENAI_API_KEY` for answer generation

The OpenAI provider uses the Responses API through the official Python SDK.
Indexing and search do not require an OpenAI API key; it is required only for
the `/answers` endpoint or the CLI `ask` command.

## Interfaces

Start the REST service with:

```bash
uvicorn api.main:app --reload
```

The REST API exposes `GET /health`, `POST /repositories/index`, `POST /search`,
and `POST /answers`. The same long-running process must index a repository
before search or answer requests because the BM25 index is process-local.

For terminal use:

```bash
python cli.py index /path/to/repository --name my-repository
python cli.py ask /path/to/repository "How is authentication implemented?"
```

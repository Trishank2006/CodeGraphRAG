![tests](https://github.com/Trishank2006/CodeGraphRAG/actions/workflows/tests.yml/badge.svg)

# CodeGraphRAG

**A code RAG system for benchmarking dense, lexical, and graph-based retrieval.**

CodeGraphRAG indexes source-code repositories, retrieves relevant code chunks, and generates answers with source-file and line-range citations. It supports dense vector search, BM25 lexical search, graph retrieval over a Neo4j code graph, reciprocal rank fusion (RRF), and optional cross-encoder reranking.

Dense retrieval is the default. On the project's 100-query benchmark, hybrid retrieval (dense + BM25 + graph, fused with RRF) scored slightly higher than dense alone: Hit@5 of 0.93 versus 0.90, with similar gains on MRR@5 and nDCG@5. That is a difference of three queries, so treat it as a modest improvement. Hybrid retrieval is one setting away; see [Retrieval configuration](#retrieval). These results apply to the tested repository and configuration, not necessarily to other codebases.

## Quickstart

Requires Python, Git, Docker, and [Ollama](https://ollama.com/).

```bash
git clone https://github.com/Trishank2006/CodeGraphRAG.git
cd CodeGraphRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then set CODEGRAPHRAG_NEO4J_PASSWORD in .env
docker compose up -d          # start Neo4j
ollama pull qwen2.5:3b        # download the local model

python cli.py ask . "Where is BM25 tokenization implemented?" --name codegraphrag
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and copy with `Copy-Item .env.example .env`.

The first run indexes the repository, then answers with citations. Later questions reuse the index. Each step is explained under [Installation](#installation).

## Features

- Tree-sitter parsing and AST-aware code chunking.
- Dense semantic retrieval with `BAAI/bge-small-en-v1.5`.
- Identifier-aware BM25 lexical retrieval that splits `snake_case`, `CamelCase`, and file paths.
- A code graph of definitions, imports, and calls, stored in Neo4j, with conservative call resolution.
- Hybrid retrieval combining dense, BM25, and graph results with reciprocal rank fusion.
- Optional cross-encoder reranking.
- LLM-generated answers with source citations, grounded in graph context from Neo4j.
- Local inference through Ollama, or any hosted OpenAI-compatible API.
- CLI and REST API for indexing repositories and asking questions.
- Queries against existing indexes without re-indexing.
- Retrieval benchmarking with per-query diagnostics.

## Architecture

For more detail on the service, indexing phases, and code graph, see [`docs/architecture.md`](docs/architecture.md).

### Indexing

```text
Source Repository
       |
       v
Repository Ingestion
       |
       v
Tree-sitter Parsing and Code Chunking
       |
       +------------------------+
       |                        |
       v                        v
Embedding Generation      Code Graph Construction
       |                        |
       v                        v
Local Qdrant Store              Neo4j
```

The parser extracts code entities and their relationships, including definitions, imports, and calls. Calls are linked to their definitions only when the target can be determined without guessing; see [Supported languages](#supported-languages).

### Default retrieval and answer generation

```text
Question
   |
   v
Dense Vector Retrieval (Qdrant)
   |
   v
Retrieved Code Context  <---  Graph Grounding (Neo4j)
   |
   v
LLM Answer Generation
   |
   v
Answer + Source Citations
```

Dense mode retrieves chunks from Qdrant only. Neo4j is still used at answer time, to add graph context to the retrieved code. Overlapping chunks from the same file are de-duplicated before they reach the LLM.

### Hybrid retrieval

```text
Dense Retrieval + BM25 + Graph Retrieval (Neo4j)
                    |
                    v
       Reciprocal Rank Fusion (RRF)
                    |
                    v
      Optional Cross-Encoder Reranking
                    |
                    v
          Retrieved Code Context
```

Select hybrid mode with `CODEGRAPHRAG_RETRIEVAL_MODE=hybrid`. The BM25 index lives in memory and is rebuilt from Qdrant automatically after a restart. The normal application path does not use reranking; see [Reranking](#reranking).

### Re-indexing safety

Indexing parses, chunks, embeds, and validates everything before writing to either store. Neo4j then replaces the repository's graph in a single transaction. After that, Qdrant writes the new points before deleting stale ones, so the repository is never empty during the swap.

If indexing fails before the writes, the previous index is untouched. If the Neo4j write fails, it rolls back and the vectors are untouched. If the Qdrant write fails after Neo4j succeeds, the graph is newer than the vectors, but the old vectors remain searchable; running `index` again restores consistency.

## Requirements

- Python. Development and the reported benchmark used Python 3.14.3.
- Git.
- Docker with Docker Compose, for running Neo4j.
- Ollama for local inference, or an API key for a hosted OpenAI-compatible provider.
- Internet access on first run to download model weights, unless they are already cached.

The development machine had 16 GB RAM and an NVIDIA RTX 3050 with 4 GB VRAM. Resource requirements depend on repository size, model choice, and inference configuration.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Trishank2006/CodeGraphRAG.git
cd CodeGraphRAG
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create your `.env` file

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Keep `.env` out of version control. Never commit API keys or production database credentials.

### 5. Set the Neo4j credentials

Set these in `.env` before starting Neo4j for the first time:

```dotenv
CODEGRAPHRAG_NEO4J_USER=neo4j
CODEGRAPHRAG_NEO4J_PASSWORD=replace-with-a-strong-password
```

`compose.yaml` reads the same variables, so this configures both the database and the application.

### 6. Start Neo4j

```bash
docker compose up -d
docker compose ps
```

The provided `compose.yaml` uses `neo4j:5.26.30`. It exposes Neo4j Browser at [http://localhost:7474](http://localhost:7474) and Bolt at `bolt://localhost:7687`.

Neo4j applies the credentials when its data volume is first initialized. Changing the password in `.env` later does not change the password in an existing database.

To start over, run:

```bash
docker compose down -v
```

**Warning:** this removes the Compose-managed volumes and deletes the graph data stored in those volumes.

## LLM configuration

Add one of the following configurations to `.env`.

### Option A: Local inference with Ollama

`.env.example` already contains the Ollama settings, so if you copied it in step 4, you only need to install [Ollama](https://ollama.com/), start it, and download the model:

```bash
ollama pull qwen2.5:3b
```

For reference, these are the settings it uses:

```dotenv
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:3b
CODEGRAPHRAG_OPENAI_MODEL=qwen2.5:3b
```

The API key `ollama` is a non-empty placeholder, not a real secret.

### Option B: Hosted inference with Groq

```dotenv
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=your-groq-api-key
OPENAI_MODEL=llama-3.3-70b-versatile
CODEGRAPHRAG_OPENAI_MODEL=llama-3.3-70b-versatile
```

Other OpenAI-compatible providers can be configured using their endpoint, API key, and a model they serve.

### How settings are resolved

- `CODEGRAPHRAG_OPENAI_MODEL` takes precedence over `OPENAI_MODEL`.
- If neither variable is set, the application defaults to `llama-3.3-70b-versatile`. The configured provider must serve that model, or you must specify a supported alternative.
- The application loads `.env` using `python-dotenv` with `override=False`. Variables already set in the process environment take precedence over values in `.env`.

## Usage

### Ask a question

```bash
python cli.py ask "/path/to/your/repository" "Where is authentication implemented?" --name my-repository
```

On Windows PowerShell, use a Windows path:

```powershell
python cli.py ask "C:\path\to\your\repository" "Where is authentication implemented?" --name my-repository
```

`ask` indexes the repository only if no index exists for that name. Otherwise it answers from the existing index. After changing the code, add `--reindex` to refresh it.

### Index without asking

```bash
python cli.py index "/path/to/your/repository" --name my-repository
```

The summary reports file, entity, chunk, graph-node, and graph-edge counts.

### Exclude directories

Both commands accept `--exclude`, which skips a directory or file relative to the repository root. It can be repeated:

```bash
python cli.py index "/path/to/your/repository" --name my-repository --exclude tests --exclude docs
```

`--exclude` only takes effect when indexing runs. With `ask`, combine it with `--reindex` if an index already exists.

### Repository names

The repository name defaults to the directory name. Indexing under an existing name replaces that repository's vector and graph data. Each repository's data is stored separately, so indexing one never affects another.

### CLI help

```bash
python cli.py --help
python cli.py index --help
python cli.py ask --help
```

## REST API

Start the API server:

```bash
uvicorn api.main:app
```

| Method | Path | Body | Purpose |
|---|---|---|---|
| `GET` | `/health` | | Liveness check |
| `POST` | `/repositories/index` | `repository_path`, optional `repository_name`, `exclude` | Index a repository |
| `POST` | `/search` | `query`, optional `repository_name` | Retrieve code chunks |
| `POST` | `/answers` | `query`, optional `repository_name` | Generate a cited answer |

Without `repository_name`, queries go to the most recently indexed repository. Interactive documentation is available at `/docs` while the server runs.

Only directories inside the allowed root can be indexed. Set it with `CODEGRAPHRAG_API_REPOSITORY_ROOT`; it defaults to the server's working directory. Paths outside it return `403`.

The API has no authentication, so keep it bound to localhost. Run it as a single process: the embedded Qdrant store can only be opened by one process at a time. `/health` reports that the server is running; it does not yet check Neo4j or Qdrant.

## Configuration

Settings are provided through environment variables and can be placed in `.env`. They are read once when the service starts.

### Retrieval

| Variable | Default | Purpose |
|---|---|---|
| `CODEGRAPHRAG_RETRIEVAL_MODE` | `dense` | Retrieval mode: `dense` or `hybrid`. |
| `CODEGRAPHRAG_DENSE_TOP_K` | `20` | Dense retrieval candidate limit. |
| `CODEGRAPHRAG_BM25_TOP_K` | `20` | BM25 candidate limit in hybrid mode. |
| `CODEGRAPHRAG_GRAPH_TOP_K` | `20` | Graph retrieval candidate limit in hybrid mode. |
| `CODEGRAPHRAG_RRF_TOP_K` | `30` | Maximum number of fused candidates. |
| `CODEGRAPHRAG_RRF_K` | `60` | Reciprocal rank fusion constant. |
| `CODEGRAPHRAG_RERANK_TOP_K` | `10` | Output limit when reranking is active. |
| `CODEGRAPHRAG_FUSION_WEIGHT` | Unset | Optional weighted rank-fusion configuration. |
| `CODEGRAPHRAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model used by the benchmark configuration. |

### Storage

| Variable | Default | Purpose |
|---|---|---|
| `CODEGRAPHRAG_QDRANT_PATH` | `data/qdrant` | Local Qdrant persistence directory. |
| `CODEGRAPHRAG_QDRANT_COLLECTION` | `code_chunks` | Qdrant collection for code chunks. |
| `CODEGRAPHRAG_NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI. |
| `CODEGRAPHRAG_NEO4J_USER` | `neo4j` | Neo4j username. |
| `CODEGRAPHRAG_NEO4J_PASSWORD` | Required | Neo4j password. |

Qdrant runs in local embedded mode, so a separate Qdrant server is not required. Keep `data/qdrant` between runs to retain stored vectors.

The embedding model `BAAI/bge-small-en-v1.5` downloads on first use if it is not already cached. If you change the embedding model, its vector size must match the Qdrant collection; indexing checks this before writing anything.

Neo4j is required for indexing and for answer generation in every retrieval mode.

### Reranking

The normal CLI path does not use a reranker. Setting `CODEGRAPHRAG_RERANKER_MODEL` alone does not enable reranking.

The benchmark runner constructs `CrossEncoderReranker` explicitly. In application code, a reranker can be passed when constructing `CodeGraphRAGService`; it applies in hybrid mode. It reranks the fused candidates (up to 30 by default) and keeps up to 10.

## Supported languages

Tree-sitter grammars are included for Python, JavaScript, TypeScript, Java, C, C++, Go, and Rust. Code entity and graph-relationship extraction is supported for these languages.

Calls are linked to a definition only when the target is unambiguous:

- **All languages:** functions defined in the same file, and helpers nested inside the calling function.
- **Python:** `self.method()` calls, class-level calls such as `Settings.from_environment()`, and functions from imported modules, including relative imports.
- **Java and C++:** bare calls to methods of the enclosing class.
- **Go and Java:** definitions in the same package (directory).

Anything else, including built-ins, third-party libraries, calls through variables, and ambiguous names, stays an unresolved placeholder rather than a guessed edge.

## Retrieval benchmark

Benchmark history, the checks performed, and the areas not yet measured are recorded in [`docs/benchmarks.md`](docs/benchmarks.md).

### Methodology

The benchmark runs 100 manually verified queries against CodeGraphRAG's own source code. The `evaluation/` and `tests/` directories are excluded from the benchmark index. The reported run indexed 57 files into 240 chunks.

Each query in [`evaluation/dataset.py`](evaluation/dataset.py) includes expected files, expected symbols, and a reference answer. The project author wrote the queries for this codebase, so this is a single-repository self-retrieval benchmark rather than an independent evaluation. All queries are lookups ("Where is X implemented?"); none ask about relationships between code.

Metrics are computed at `k = 5`.

### Results

| Retrieval strategy | Hit@5 | Precision@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9000 | **0.2000** | 0.8950 | 0.7564 | 0.7880 |
| BM25 | 0.8900 | 0.1800 | 0.8900 | 0.7504 | 0.7774 |
| Dense + BM25 (RRF) | **0.9300** | 0.1900 | **0.9250** | 0.7924 | 0.8202 |
| Dense + BM25 + Graph | **0.9300** | 0.1900 | **0.9250** | **0.7926** | **0.8222** |
| Dense + BM25 + Graph + Reranker | 0.8100 | 0.2322 | 0.8050 | 0.5823 | 0.6387 |

The reranked configuration has the highest Precision@5 but ranks worse on every other metric.

### Metric definitions

- **Hit@5:** fraction of queries with at least one relevant result in the top five.
- **Precision@5:** fraction of the top five results that are relevant.
- **Recall@5:** fraction of a query's expected targets found in the top five.
- **MRR@5:** reciprocal rank of the first relevant result, or zero if none appears in the top five, averaged across queries.
- **nDCG@5:** ranking-quality metric that rewards relevant results appearing higher in the list.

Each query's expected files and symbols define relevance. The dataset averages 1.03 expected files and 0.98 expected symbols per query, so Precision@5 is low by construction. Read it alongside Hit@5 and Recall@5.

### Findings

- **Hybrid retrieval scored highest.** Fusing dense and BM25 results with RRF reached Hit@5 0.93, MRR@5 0.792, and nDCG@5 0.820, compared with 0.90, 0.756, and 0.788 for dense alone. The Hit@5 difference is three queries out of 100.
- **Identifier-aware tokenization mattered most.** An earlier version of this benchmark, before BM25 split `CamelCase` identifiers, measured BM25 at Hit@5 0.74 and MRR@5 0.577. With the fix, BM25 reaches 0.89 and 0.750. The corpus differed slightly between runs (201 chunks then, 240 now), so treat this comparison as approximate.
- **Graph retrieval was neutral.** Adding it to RRF changed MRR@5 by 0.0002 and nDCG@5 by 0.002. Lookup queries don't need code relationships, so this benchmark can't measure what graph retrieval is designed for.
- **The general-purpose reranker hurt ranking.** The `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker lowered Hit@5 to 0.81 and MRR@5 to 0.582. It is not used in the normal application path. A code-specific reranker is a candidate for future evaluation.

These findings apply to this dataset and configuration. Results may differ on other repositories, languages, and query sets.

### Reproduce the benchmark

```bash
python -m evaluation.benchmark_runner --repository . --repository-name codegraphrag-benchmark --top-k 5 --output evaluation/results/benchmark-local.json
```

Use a dedicated repository name so the benchmark does not replace another index. This command evaluates retrieval only. To evaluate answer generation as well, configure an LLM and add `--evaluate-generation`.

The reported results do not establish answer-generation quality, faithfulness, or citation accuracy.

## Example output

The following output was captured from a local run using Qwen2.5 3B.

**Question**

> Where is the dense retrieval default configured?

**Command**

```bash
python cli.py ask "." "Where is the dense retrieval default configured?" --name codegraphrag-local-test --reindex
```

**Indexing summary**

```text
Repository: codegraphrag-local-test
Files: 57
Entities: 235
Chunks: 240
Graph nodes: 574
Graph edges: 1168
```

**Generated answer**

> The dense retrieval default is configured in the `application/service.py` file at line 63. [8]

**Citations returned**

1. `application/service.py`: `_dense_search`, lines 268–278
2. `retrieval/pipeline.py`: `__init__`, lines 82–99
3. `retrieval/hybrid.py`: `__init__`, lines 22–32
4. `retrieval/config.py`: `RetrievalConfig`, lines 7–73
5. `application/service.py`: `search`, lines 280–305
6. `retrieval/search.py`: `search_code`, lines 6–64
7. `retrieval/hybrid.py`: `search`, lines 34–88
8. `application/service.py`: `__init__`, lines 61–104
9. `retrieval/pipeline.py`: `search`, lines 101–136
10. `vector_store/qdrant_store.py`: `search`, lines 173–238

**Manual correctness check:** The answer is partly correct. Citation [8] points to the right place: `CodeGraphRAGService.__init__` reads `CODEGRAPHRAG_RETRIEVAL_MODE` and falls back to `"dense"`. The line number is wrong, though. The fallback is on line 88; line 63 is the method's `settings` parameter.

No two citations overlap: overlapping chunks from the same file are removed before the context reaches the LLM.

This example shows why generated answers and citations require verification. Generation and citation quality have not been formally evaluated beyond this manual check.

## Project layout

| Path | Purpose |
|---|---|
| `api/` | FastAPI REST API |
| `application/` | Application configuration and service orchestration |
| `ingestion/` | Repository file discovery, language detection, and ingestion |
| `parser/` | Tree-sitter parsing and code entity/relationship extraction |
| `chunking/` | Code chunk construction |
| `embeddings/` | Embedding model integration |
| `vector_store/` | Qdrant persistence and indexing |
| `graph/` | Graph construction, call resolution, Neo4j storage, traversal, and retrieval |
| `retrieval/` | Dense, BM25, hybrid, fusion, and reranking components |
| `generation/` | Context construction, prompting, and answer generation |
| `evaluation/` | Datasets, benchmark runners, metrics, and results |
| `docs/` | Architecture notes (`architecture.md`) and benchmark notes (`benchmarks.md`) |
| `scripts/` | Utility scripts |
| `tests/` | Automated tests |
| `.github/workflows/` | Continuous integration |
| `cli.py` | Command-line entry point |
| `compose.yaml` | Local Neo4j service |
| `.env.example` | Example environment configuration |

## Testing

Run the test suite with:

```bash
python -m pytest -rs -q
```

The `-rs` flag prints the reason for each skipped test. GitHub Actions runs the suite on every pull request and on every push to `main`.

Two tests are opt-in live integration tests, which are skipped unless enabled:

- `tests/test_live_neo4j.py` requires `CODEGRAPHRAG_RUN_LIVE_TESTS=1` and a running Neo4j instance.
- `tests/test_live_end_to_end.py` requires `CODEGRAPHRAG_RUN_LIVE_TESTS=1`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL`, plus a running Neo4j instance and LLM endpoint.

Both live tests have passed against a local Neo4j instance and a local Ollama model. CI runs without live services, so it skips them.

## Limitations and future work

- **Benchmark scope:** The retrieval benchmark covers one repository and 100 author-written lookup queries. It does not ask relational questions such as "what calls this function?", so it can't show the value of graph retrieval. Adding relational queries and other repositories is the next evaluation step.
- **Generation evaluation:** Answer quality, faithfulness, citation validity, and citation coverage have not been formally evaluated.
- **Cross-file call resolution:** Python imports and Go and Java packages are resolved. JavaScript, TypeScript, Rust, C, and C++ cross-file calls remain unresolved placeholders, because the parser does not yet capture imports for those languages.
- **Neo4j dependency:** Neo4j is required for indexing and answer generation, even in dense retrieval mode.
- **Reranking:** The tested general-purpose cross-encoder reduced ranking quality. A code-specific reranker needs evaluation.
- **Stale indexes:** `ask` reuses an existing index. After code changes, use `--reindex`.
- **API deployment:** The API has no authentication, must run as a single process, and its health check does not yet verify Neo4j or Qdrant.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for the full text.
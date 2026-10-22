# CodeGraphRAG

**A code RAG system for benchmarking dense, lexical, and graph-based retrieval.**

CodeGraphRAG indexes source-code repositories, retrieves relevant code chunks, and generates answers with source-file and line-range citations. It supports dense vector search, BM25 lexical search, graph retrieval over a Neo4j code graph, reciprocal rank fusion (RRF), and optional cross-encoder reranking.

Dense retrieval is the default. On the project's 100-query benchmark, it outperformed every tested combined pipeline on Hit@5, Recall@5, MRR@5, and nDCG@5. These results apply to the tested repository and configuration, not necessarily to other codebases.

## Features

- Tree-sitter parsing and AST-aware code chunking.
- Dense semantic retrieval with `BAAI/bge-small-en-v1.5`.
- BM25 lexical retrieval.
- A code graph of definitions, imports, and calls, stored in Neo4j.
- Hybrid retrieval combining dense, BM25, and graph results with reciprocal rank fusion.
- Optional cross-encoder reranking.
- LLM-generated answers with source citations.
- Local inference through Ollama or hosted OpenAI-compatible APIs.
- CLI for indexing repositories and asking questions.
- Retrieval benchmarking with per-query diagnostics.

## Architecture

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

The parser extracts code entities and their relationships, including definitions, imports, and calls.

### Default retrieval: dense

```text
Question
   |
   v
Dense Vector Retrieval (Qdrant)
   |
   v
Retrieved Code Context
   |
   v
LLM Answer Generation
   |
   v
Answer + Source Citations
```

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

Select hybrid mode with `CODEGRAPHRAG_RETRIEVAL_MODE=hybrid`.

The normal application path does not use reranking by default. See [Reranking](#reranking) for details.

## Requirements

- Python. Development and the reported benchmark used Python 3.14.3.
- Git.
- Docker with Docker Compose, for running Neo4j.
- Ollama for local inference, or an API key for a hosted OpenAI-compatible provider.
- Internet access on first run to download model weights, unless they are already cached.

The development machine had 16 GB RAM and an NVIDIA RTX 3050 with 4 GB VRAM. Resource requirements depend on repository size, model choice, and the inference configuration.

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

Install [Ollama](https://ollama.com/), start it, and download a model:

```bash
ollama pull qwen2.5:3b
```

Configure the OpenAI-compatible API endpoint:

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

The `ask` command indexes the repository before answering, so you do not need to run `index` first. However, each invocation repeats the indexing work, which can add latency for larger repositories.

### Index without asking

```bash
python cli.py index "/path/to/your/repository" --name my-repository
```

The summary reports file, entity, chunk, graph-node, and graph-edge counts.

### Repository names

Indexing under an existing repository name replaces that repository's vector and graph data once new chunks have been generated successfully. If no chunks are generated, the existing index is preserved.

Use a distinct repository name for each separate experiment to avoid unintentionally replacing another index.

### CLI help

```bash
python cli.py --help
python cli.py index --help
python cli.py ask --help
```

## Configuration

Settings are provided through environment variables and can be placed in `.env`.

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

Qdrant runs in local embedded mode, so a separate Qdrant server is not required by the current configuration. Keep `data/qdrant` between runs to retain stored vectors.

The embedding model `BAAI/bge-small-en-v1.5` downloads on first use if it is not already cached.

Neo4j is required during service initialization and indexing, even in dense mode, although the dense retrieval path does not query the graph for retrieval.

## Reranking

The normal CLI path does not use a reranker by default. Setting `CODEGRAPHRAG_RERANKER_MODEL` alone does not enable reranking.

The benchmark runner constructs `CrossEncoderReranker` explicitly. In application code, a reranker can also be passed when constructing `CodeGraphRAGService`.

When active, the benchmark configuration reranks the fused candidate pool, which contains up to 30 candidates by default, and retains up to 10. The evaluation metrics are then computed at the top five results.

## Supported languages

Tree-sitter grammars are included for:

- Python
- JavaScript
- TypeScript
- Java
- C
- C++
- Go
- Rust

Code entity and graph-relationship extraction is supported for these languages.

## Retrieval benchmark

### Methodology

The benchmark runs 100 manually verified queries against CodeGraphRAG's own source code. The reported run indexed 57 files into 201 chunks.

Each query in [`evaluation/dataset.py`](evaluation/dataset.py) includes expected files, expected symbols, and a reference answer. The project author wrote the queries for this codebase, so this is a single-repository self-retrieval benchmark rather than an independent evaluation.

Metrics are computed at `k = 5`.

### Results

| Retrieval strategy | Hit@5 | Precision@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Dense | **0.9000** | 0.1840 | **0.8950** | **0.7647** | **0.7887** |
| BM25 | 0.7400 | 0.1500 | 0.7350 | 0.5766 | 0.6051 |
| Dense + BM25 (RRF) | 0.8300 | 0.1700 | 0.8250 | 0.7032 | 0.7203 |
| Dense + BM25 + Graph | 0.8300 | 0.1700 | 0.8250 | 0.7020 | 0.7203 |
| Dense + BM25 + Graph + Reranker | 0.8200 | **0.1847** | 0.8150 | 0.5862 | 0.6364 |

### Metric definitions

- **Hit@5:** fraction of queries with at least one relevant result in the top five.
- **Precision@5:** fraction of the top five results that are relevant.
- **Recall@5:** fraction of a query's expected targets found in the top five.
- **MRR@5:** reciprocal rank of the first relevant result, or zero if none appears in the top five, averaged across queries.
- **nDCG@5:** ranking-quality metric that rewards relevant results appearing higher in the list.

Each query's expected files and symbols define relevance. The dataset averages 1.03 expected files and 0.98 expected symbols per query. Because each query has a relatively small set of expected targets, Precision@5 should be interpreted alongside Hit@5 and Recall@5.

### Findings

- Dense retrieval achieved the highest Hit@5, Recall@5, MRR@5, and nDCG@5 among the tested configurations.
- Adding BM25 through RRF lowered Hit@5 from 0.90 to 0.83.
- Adding graph retrieval made no difference to Hit@5, Recall@5, or nDCG@5 compared with Dense + BM25. MRR@5 changed from 0.7032 to 0.7020.
- The tested general-purpose reranker lowered MRR@5 from 0.7020 to 0.5862. It reranked up to 30 fused candidates and retained 10, with metrics computed on the top five.
- The reranked configuration had Precision@5 of 0.1847, compared with 0.1840 for dense retrieval. This small difference does not offset the lower scores on the other reported ranking metrics.
- Dense retrieval is therefore the default. A code-specific reranker is a candidate for future evaluation.

These findings apply to this dataset and configuration. Results may differ on other repositories, languages, and query sets.

### Reproduce the benchmark

Use a dedicated repository name so the benchmark does not replace another index:

```bash
python -m evaluation.benchmark_runner --repository . --repository-name codegraphrag-benchmark --top-k 5 --output evaluation/results/benchmark-local.json
```

This command evaluates retrieval only. To evaluate answer generation as well, configure an LLM and add `--evaluate-generation`.

The reported results do not establish answer-generation quality, faithfulness, or citation accuracy. For a meaningful comparison, keep the dataset, retrieval configuration, model settings, and evaluation procedure consistent across runs.

## Example output

The following output was captured from a local run using Qwen2.5 3B.

**Question**

> Where is the dense retrieval default configured?

**Command**

```bash
python cli.py ask "." "Where is the dense retrieval default configured?" --name codegraphrag-local-test
```

**Indexing summary**

```text
Repository:   codegraphrag-local-test
Files:        57
Entities:     196
Chunks:       201
Graph nodes:  570
Graph edges:  1275
```

**Generated answer**

> The dense retrieval default is configured in the `application/service.py` file at lines 174–181. The configuration specifies a `top_k` value of 20 by default.

**Citations returned**

- `retrieval/pipeline.py` — `__init__`, lines 82–99
- `retrieval/hybrid.py` — `__init__`, lines 22–32
- `application/service.py` — `dense_search`, lines 174–181
- `application/service.py` — `search`, lines 170–212
- `retrieval/config.py` — `RetrievalConfig`, lines 7–73

**Manual correctness check:** The answer is partly correct. The default retrieval mode (`dense`) is selected in `application/service.py`. However, the default `top_k` value of 20 is defined in `retrieval/config.py`, not in `service.py`. That configuration file appears in the returned citations, but the model attributed the value to the wrong file.

This example demonstrates why generated answers and citations require verification. Generation and citation quality have not been formally evaluated beyond this manual check.

## Project layout

| Path | Purpose |
|---|---|
| `application/` | Application configuration and service orchestration |
| `ingestion/` | Repository file discovery, language detection, and ingestion |
| `parser/` | Tree-sitter parsing and code entity/relationship extraction |
| `chunking/` | Code chunk construction |
| `embeddings/` | Embedding model integration |
| `vector_store/` | Qdrant persistence and indexing |
| `graph/` | Graph construction, Neo4j storage, traversal, and retrieval |
| `retrieval/` | Dense, BM25, hybrid, fusion, and reranking components |
| `generation/` | Context construction, prompting, and answer generation |
| `evaluation/` | Dataset, benchmark runners, metrics, and results |
| `tests/` | Automated tests |
| `cli.py` | Command-line entry point |
| `compose.yaml` | Local Neo4j service |
| `.env.example` | Example environment configuration |

## Testing

Run the test suite with:

```bash
python -m pytest -rs -q
```

The `-rs` flag prints the reason for each skipped test.

The latest reported development run passed **324 tests, with 2 skipped**. Both skipped tests are opt-in live integration tests:

- `tests/test_live_end_to_end.py` requires `CODEGRAPHRAG_RUN_LIVE_TESTS=1`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL`.
- `tests/test_live_neo4j.py` requires `CODEGRAPHRAG_RUN_LIVE_TESTS=1` and a running Neo4j instance.

Test counts describe that development run and may change as the codebase evolves.

## Limitations and future work

- **Index reuse:** `ask` re-indexes the repository before every answer. Querying an existing index without re-indexing is not yet supported by this workflow.
- **Neo4j dependency:** Neo4j is required for service initialization and indexing, even in dense mode.
- **Generation evaluation:** Answer quality, faithfulness, citation validity, and citation coverage have not been formally evaluated.
- **Reranking:** The tested general-purpose cross-encoder reduced ranking quality on most reported metrics. A code-specific reranker needs evaluation.
- **Benchmark scope:** The benchmark covers one repository and 100 author-written queries. Broader evaluation is needed before generalizing the findings.
- **Language coverage:** Code entity and relationship extraction is limited to the listed Tree-sitter-supported languages.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for the full text.
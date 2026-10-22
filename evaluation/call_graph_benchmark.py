"""Call-graph resolution benchmark.

Compares the CALLS edges that CodeGraphRAG stores in Neo4j against
hand-labeled relationships in this repository's own source code. Each case
names one function and lists the repository definitions it calls (callees)
or the definitions that call it (callers).

Run from the repository root with Neo4j running:

    python -m evaluation.call_graph_benchmark

The labels were written by reading the source code. When the code changes, a
mismatch can mean a resolver error or an outdated label, so every case lists
its missing and unexpected edges for review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY = "codegraphrag-callgraph"
EXCLUDE = ("evaluation", "tests")

SERVICE = "application/service.py"
BM25 = "retrieval/bm25.py"
QDRANT = "vector_store/qdrant_store.py"
BUILDER = "graph/builder.py"
CLI = "cli.py"

Edge = tuple[str, str]


@dataclass(frozen=True)
class Case:
    direction: str  # "callees" or "callers"
    file_path: str
    name: str
    expected: frozenset[Edge]
    kind: str = "function"


def callees(file_path: str, name: str, *expected: Edge) -> Case:
    return Case("callees", file_path, name, frozenset(expected))


def callers(file_path: str, name: str, *expected: Edge) -> Case:
    return Case("callers", file_path, name, frozenset(expected))


CASES = [
    # What a function calls.
    callees(
        SERVICE, "index_repository",
        ("_normalize_exclusions", SERVICE),
        ("discover_files", "ingestion/file_discovery.py"),
        ("_is_excluded", SERVICE),
        ("extract_metadata", "ingestion/metadata.py"),
        ("detect_language", "ingestion/language_detection.py"),
        ("parse_source_code", "parser/core_parser.py"),
        ("build_repository_chunks", "vector_store/repository_indexer.py"),
        ("NoChunksGeneratedError", SERVICE),
        ("GraphBuilder", BUILDER),
        ("embed_chunks", "vector_store/indexer.py"),
        ("_validate_before_write", SERVICE),
        ("_bm25_instance_for", SERVICE),
        ("IndexSummary", SERVICE),
    ),
    callees(
        SERVICE, "search",
        ("_require_index", SERVICE),
        ("GraphRetriever", "graph/retrieval.py"),
        ("HybridRetrievalPipeline", "retrieval/pipeline.py"),
        ("_bm25_for", SERVICE),
    ),
    callees(
        SERVICE, "answer",
        ("_require_index", SERVICE),
        ("GraphGroundingService", "graph/grounding.py"),
        ("AnswerGenerator", "generation/generator.py"),
        ("_RepositoryRetriever", SERVICE),
        ("ContextBuilder", "generation/context.py"),
        ("PromptBuilder", "generation/prompt.py"),
        ("_get_llm_client", SERVICE),
    ),
    callees(SERVICE, "check_dependencies", ("check_graph_health", "graph/health.py")),
    callees(SERVICE, "_drop_nested_results", ("_nested", SERVICE)),
    callees("graph/health.py", "check_graph_health", ("Neo4jStore", "graph/neo4j_store.py")),
    callees(BM25, "index", ("_tokenize", BM25)),
    callees(
        BM25, "search_bm25",
        ("_tokenize", BM25),
        ("_score_document", BM25),
        ("RetrievalResult", "retrieval/models.py"),
    ),
    callees(BM25, "_score_document", ("_score_field", BM25)),
    callees(BM25, "_score_field", ("_field_idf", BM25)),
    callees(
        QDRANT, "replace_repository",
        ("_validate_repository", QDRANT),
        ("upsert_chunks", QDRANT),
        ("_repository_condition", QDRANT),
        ("point_id_for", QDRANT),
    ),
    callees(QDRANT, "upsert_chunks", ("point_id_for", QDRANT)),
    callees(
        QDRANT, "count_repository",
        ("_collection_exists", QDRANT),
        ("_repository_condition", QDRANT),
    ),
    callees(
        BUILDER, "build_from_parsed_files",
        ("build", BUILDER),
        ("add_node", BUILDER),
        ("add_reference_node", BUILDER),
        ("GraphNode", "graph/models.py"),
        ("_scoped_id", BUILDER),
        ("RelationType", "graph/schema.py"),
        ("_absolute_module", BUILDER),
        ("GraphEdge", "graph/models.py"),
    ),
    callees(
        CLI, "main",
        ("build_parser", CLI),
        ("CodeGraphRAGService", SERVICE),
        ("_print_index_summary", CLI),
        ("_print_answer", CLI),
    ),
    # What calls a function.
    callers(BM25, "_tokenize", ("index", BM25), ("search_bm25", BM25)),
    callers(
        QDRANT, "point_id_for",
        ("upsert_chunks", QDRANT),
        ("replace_repository", QDRANT),
    ),
    callers(
        QDRANT, "_repository_condition",
        ("delete_repository", QDRANT),
        ("replace_repository", QDRANT),
        ("count_repository", QDRANT),
        ("load_repository_chunks", QDRANT),
        ("search", QDRANT),
    ),
    callers(SERVICE, "_validate_before_write", ("index_repository", SERVICE)),
    callers(
        BUILDER, "_absolute_module",
        ("build", BUILDER),
        ("build_from_parsed_files", BUILDER),
    ),
    callers(SERVICE, "_nested", ("_drop_nested_results", SERVICE)),
    callers(
        SERVICE, "_require_index",
        ("has_repository", SERVICE),
        ("search", SERVICE),
        ("answer", SERVICE),
    ),
    callers(
        SERVICE, "_bm25_instance_for",
        ("index_repository", SERVICE),
        ("_bm25_for", SERVICE),
    ),
]

CALLEES_QUERY = """
MATCH (:Node {id: $id})-[r:RELATION]->(other:Node)
WHERE r.type = 'CALLS'
RETURN other.name AS name, other.file_path AS file_path,
       coalesce(other.reference, false) AS reference
"""

CALLERS_QUERY = """
MATCH (other:Node)-[r:RELATION]->(:Node {id: $id})
WHERE r.type = 'CALLS'
RETURN other.name AS name, other.file_path AS file_path,
       coalesce(other.reference, false) AS reference
"""

NODE_EXISTS_QUERY = "MATCH (n:Node {id: $id}) RETURN count(n) AS count"


def node_id(case: Case) -> str:
    return f"repo:{REPOSITORY}:{case.file_path}:{case.kind}:{case.name}"


def _pairs(edges) -> list[list[str]]:
    return [list(edge) for edge in sorted(edges)]


def evaluate(store, case: Case) -> dict[str, Any]:
    """Compare one case's stored edges with its labels."""
    result: dict[str, Any] = {
        "direction": case.direction,
        "entity": f"{case.file_path}::{case.name}",
        "expected": _pairs(case.expected),
    }

    exists = store.execute_query(NODE_EXISTS_QUERY, {"id": node_id(case)})
    if not exists or not exists[0].get("count"):
        result.update(
            error="node not found",
            found=[],
            missing=_pairs(case.expected),
            unexpected=[],
            unresolved=0,
        )
        return result

    query = CALLEES_QUERY if case.direction == "callees" else CALLERS_QUERY
    rows = store.execute_query(query, {"id": node_id(case)})

    resolved = {
        (str(row.get("name") or ""), str(row.get("file_path")))
        for row in rows
        if not row.get("reference") and row.get("file_path")
    }


    unresolved_targets = [
        {
            "name": str(row.get("name") or ""),
            "file_path": row.get("file_path"),
            "reference": bool(row.get("reference")),
        }
        for row in rows
        if row.get("reference") or not row.get("file_path")
    ]

    result.update(
        found=_pairs(resolved & case.expected),
        missing=_pairs(case.expected - resolved),
        unexpected=_pairs(resolved - case.expected),
        unresolved=len(unresolved_targets),
        unresolved_targets=unresolved_targets,
    )

    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-averaged precision and recall over all labeled edges."""
    expected = sum(len(r["expected"]) for r in results)
    found = sum(len(r["found"]) for r in results)
    resolved = found + sum(len(r["unexpected"]) for r in results)

    return {
        "cases": len(results),
        "expected_edges": expected,
        "found_edges": found,
        "resolved_edges": resolved,
        "precision": round(found / resolved, 4) if resolved else 0.0,
        "recall": round(found / expected, 4) if expected else 0.0,
        "unresolved_targets": sum(
            r["unresolved"] for r in results if r["direction"] == "callees"
        ),
        "missing_nodes": sum(1 for r in results if r.get("error")),
    }


def print_report(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    for r in results:
        if r.get("error"):
            status = "ERROR"
        elif r["missing"] or r["unexpected"]:
            status = "DIFF"
        else:
            status = "OK"

        print(
            f"[{status:5}] {r['direction']:7} {r['entity']}  "
            f"found {len(r['found'])}/{len(r['expected'])}"
        )
        if r.get("error"):
            print(f"         {r['error']}")
        for name, path in r["missing"]:
            print(f"         missing:    {name} ({path})")
        for name, path in r["unexpected"]:
            print(f"         unexpected: {name} ({path})")

    print()
    print(f"Cases: {summary['cases']}   Labeled edges: {summary['expected_edges']}")
    print(f"Precision: {summary['precision']:.4f}   Recall: {summary['recall']:.4f}")
    print(f"Unresolved call targets (placeholders): {summary['unresolved_targets']}")


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call-graph resolution benchmark")
    parser.add_argument("--repository", default=".", help="Repository root to index")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Reuse the existing index instead of re-indexing",
    )
    parser.add_argument("--output", default="evaluation/results/call-graph.json")
    args = parser.parse_args(argv)

    from application.service import CodeGraphRAGService

    service = CodeGraphRAGService()
    try:
        if not args.skip_index:
            summary = service.index_repository(
                args.repository, REPOSITORY, exclude=EXCLUDE
            )
            print(
                f"Indexed {summary.files} files into {summary.graph_nodes} nodes "
                f"and {summary.graph_edges} edges."
            )
            print()
        results = [evaluate(service.graph_store, case) for case in CASES]
    finally:
        service.close()

    totals = summarize(results)
    print_report(results, totals)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "commit": _commit(),
                "summary": totals,
                "cases": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"Results saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations



from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from application.service import CodeGraphRAGService
from evaluation.latency import calculate_latency_stats, latency_stats_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CodeGraphRAG indexing and retrieval scaling."
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--repository-name", default=None)
    parser.add_argument("--queries", nargs="*", default=[
        "How does repository indexing work?",
        "Where are embeddings generated?",
        "How does hybrid retrieval combine results?",
        "How are graph relationships stored?",
        "How does BM25 search work?",
    ])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--output",
        default="evaluation/results/scaling_baseline.json",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")

    service = CodeGraphRAGService()
    try:
        stages = {}
        start = time.perf_counter()
        summary = service.index_repository(
            args.repository,
            repository_name=args.repository_name,
        )
        stages["total_indexing_ms"] = (time.perf_counter() - start) * 1000

        query_durations = []
        for query in args.queries:
            # One warm-up run; excluded from measured latency.
            service.search(query)

            durations = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                service.search(query)
                durations.append((time.perf_counter() - start) * 1000)

            query_durations.extend(durations)

        stats = calculate_latency_stats(query_durations)
        total_seconds = stages["total_indexing_ms"] / 1000

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repository_path": str(Path(args.repository).resolve()),
            "repository_name": summary.repository,
            "indexing": {
                "total_ms": round(stages["total_indexing_ms"], 3),
                "files": summary.files,
                "entities": summary.entities,
                "chunks": summary.chunks,
                "graph_nodes": summary.graph_nodes,
                "graph_edges": summary.graph_edges,
                "files_per_second": round(
                    summary.files / total_seconds, 3
                ) if total_seconds else 0,
                "chunks_per_second": round(
                    summary.chunks / total_seconds, 3
                ) if total_seconds else 0,
                "note": (
                    "Stage-level timings will be added after this "
                    "end-to-end baseline is captured."
                ),
            },
            "retrieval": {
                **latency_stats_to_dict(stats),
                "query_count": len(args.queries),
                "iterations_per_query": args.iterations,
            },
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n=== CodeGraphRAG Scaling Baseline ===")
        print(f"Repository : {summary.repository}")
        print(f"Files      : {summary.files}")
        print(f"Chunks     : {summary.chunks}")
        print(f"Graph      : {summary.graph_nodes} nodes, {summary.graph_edges} edges")
        print(f"Index time : {stages['total_indexing_ms'] / 1000:.2f} seconds")
        print(f"Throughput : {report['indexing']['chunks_per_second']} chunks/sec")
        print(f"Query p50  : {stats.p50_ms:.2f} ms")
        print(f"Query p95  : {stats.p95_ms:.2f} ms")
        print(f"Saved      : {output.resolve()}")

    finally:
        service.close()


if __name__ == "__main__":
    main()



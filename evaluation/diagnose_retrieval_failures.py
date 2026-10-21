from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from application.service import CodeGraphRAGService
from evaluation.dataset import create_default_dataset
from evaluation.service_retriever import ServiceRetriever
from retrieval.config import RetrievalConfig


def get_value(result, key):
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def result_to_dict(result):
    if isinstance(result, str):
        return {"result": result}

    return {
        "file_path": get_value(result, "file_path"),
        "symbol": get_value(result, "symbol"),
        "id": get_value(result, "id"),
        "score": get_value(result, "score"),
    }


def main():
    service = CodeGraphRAGService()

    try:
        # Use the existing isolated benchmark stores.
        service._is_indexed = True
        dataset = create_default_dataset()
        report = {
            "dataset_size": len(dataset),
            "top_k": 5,
            "candidate_count_comparison": {},
            "note": (
                "Raw top-five retrieval results. Expected files and "
                "symbols are displayed separately to help diagnose "
                "retrieval failures and relevance-label mismatches."
            ),
        }

        for candidate_count in (10, 20, 30):
            config = RetrievalConfig(
                dense_top_k=20,
                bm25_top_k=20,
                graph_top_k=20,
                rrf_top_k=candidate_count,
                rerank_top_k=5,
            )

            adapter = ServiceRetriever(service, config=config)
            retriever = adapter.as_retrievers()[
                "Dense + BM25 + Graph + Reranker"
            ]

            # Warm up the model before collecting diagnostic results.
            first_example = next(iter(dataset))
            retriever(first_example)

            query_results = []

            for example in dataset:
                raw_results = retriever(example)
                top_results = list(raw_results)[:5]

                query_results.append({
                    "query": example.query,
                    "query_type": example.query_type,
                    "expected_files": list(example.expected_files),
                    "expected_symbols": list(example.expected_symbols),
                    "results": [
                        result_to_dict(result)
                        for result in top_results
                    ],
                })

            report["candidate_count_comparison"][
                str(candidate_count)
            ] = query_results

        output = Path(
            "evaluation/results/retrieval_failure_diagnostics.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )

        print(f"Dataset queries: {len(dataset)}")
        print("Candidate counts: 10, 20, 30")
        print(f"Saved: {output.resolve()}")

        # Print symbol and conceptual queries for quick inspection.
        for candidate_count, queries in (
            report["candidate_count_comparison"].items()
        ):
            print(f"\n=== Candidates: {candidate_count} ===")

            for item in queries:
                if item["query_type"] in {
                    "symbol_lookup",
                    "conceptual",
                }:
                    print(json.dumps(item, indent=2, default=str))

    finally:
        service.close()


if __name__ == "__main__":
    main()

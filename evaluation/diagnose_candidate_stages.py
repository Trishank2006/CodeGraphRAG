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


def serialize_result(result):
    return {
        "file_path": getattr(result, "file_path", None),
        "symbol": getattr(result, "symbol", None),
        "id": getattr(result, "id", None),
        "score": getattr(result, "score", None),
    }


def matches_expected(result, example):
    file_path = getattr(result, "file_path", None)
    symbol = getattr(result, "symbol", None)

    expected_files = {
        path.replace("\\", "/")
        for path in example.expected_files
    }
    expected_symbols = set(example.expected_symbols)

    file_match = (
        file_path.replace("\\", "/") in expected_files
        if file_path
        else False
    )
    symbol_match = symbol in expected_symbols if symbol else False

    if example.query_type == "symbol_lookup":
        matched = symbol_match
    elif example.query_type == "implementation_lookup":
        matched = file_match
    else:
        matched = file_match or symbol_match

    return {
        "file_match": file_match,
        "symbol_match": symbol_match,
        "matched_under_current_rules": matched,
    }


def main():
    service = CodeGraphRAGService()

    try:
        service._is_indexed = True
        dataset = create_default_dataset()

        report = {
            "dataset_size": len(dataset),
            "candidate_count_comparison": {},
            "note": (
                "RRF candidates are captured before reranking. "
                "Final results are captured after reranking. "
                "Matches are checked against the dataset's existing "
                "relevance rules."
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
            pipeline = adapter._hybrid_pipeline(
                include_graph=True,
                include_reranker=True,
            )

            # Warm up the cross-encoder.
            first_example = next(iter(dataset))
            pipeline.search(first_example.query)

            query_reports = []

            for example in dataset:
                candidates = pipeline.hybrid.search(
                    query=example.query,
                    dense_top_k=config.dense_top_k,
                    bm25_top_k=config.bm25_top_k,
                    graph_top_k=config.graph_top_k,
                    rrf_top_k=config.rrf_top_k,
                )

                final_results = pipeline.reranker.rerank(
                    query=example.query,
                    results=candidates,
                    top_k=config.rerank_top_k,
                )

                candidate_matches = [
                    matches_expected(result, example)
                    for result in candidates
                ]
                final_matches = [
                    matches_expected(result, example)
                    for result in final_results
                ]

                query_reports.append({
                    "query": example.query,
                    "query_type": example.query_type,
                    "expected_files": list(example.expected_files),
                    "expected_symbols": list(example.expected_symbols),
                    "candidate_count": len(candidates),
                    "relevant_in_candidate_pool": any(
                        item["matched_under_current_rules"]
                        for item in candidate_matches
                    ),
                    "relevant_in_final_top5": any(
                        item["matched_under_current_rules"]
                        for item in final_matches
                    ),
                    "candidates_before_reranking": [
                        {
                            **serialize_result(result),
                            **matches_expected(result, example),
                        }
                        for result in candidates
                    ],
                    "final_results_after_reranking": [
                        {
                            **serialize_result(result),
                            **matches_expected(result, example),
                        }
                        for result in final_results
                    ],
                })

            report["candidate_count_comparison"][
                str(candidate_count)
            ] = query_reports

            total = len(query_reports)
            candidate_hits = sum(
                item["relevant_in_candidate_pool"]
                for item in query_reports
            )
            final_hits = sum(
                item["relevant_in_final_top5"]
                for item in query_reports
            )

            print(
                f"Candidates={candidate_count}: "
                f"pool hits={candidate_hits}/{total}, "
                f"final top-5 hits={final_hits}/{total}"
            )

        output = Path(
            "evaluation/results/candidate_stage_diagnostics.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )

        print(f"Saved: {output.resolve()}")

    finally:
        service.close()


if __name__ == "__main__":
    main()

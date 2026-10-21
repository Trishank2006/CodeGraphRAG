import json
from pathlib import Path

path = Path("evaluation/results/candidate_stage_diagnostics.json")
data = json.loads(path.read_text(encoding="utf-8"))
comparison = data["candidate_count_comparison"]

for count in ("10", "20", "30"):
    reports = comparison[count]

    lost = [
        item for item in reports
        if item["relevant_in_candidate_pool"]
        and not item["relevant_in_final_top5"]
    ]

    print(f"\n{'=' * 75}")
    print(f"CANDIDATES={count}: {len(lost)} queries lost during reranking")

    for item in lost:
        print(f"\nQuery: {item['query']}")
        print(f"Type: {item['query_type']}")
        print(f"Expected files: {item.get('expected_files', [])}")
        print(f"Expected symbols: {item.get('expected_symbols', [])}")

        print("\nCandidates before reranking:")
        for rank, result in enumerate(item["candidates_before_reranking"], 1):
            print(
                f"  {rank}. {result.get('file_path')} "
                f"| {result.get('symbol')} "
                f"| score={result.get('score')} "
                f"| match={result.get('matched_under_current_rules')}"
            )

        print("\nFinal top 5:")
        for rank, result in enumerate(item["final_results_after_reranking"], 1):
            print(
                f"  {rank}. {result.get('file_path')} "
                f"| {result.get('symbol')} "
                f"| score={result.get('score')} "
                f"| match={result.get('matched_under_current_rules')}"
            )

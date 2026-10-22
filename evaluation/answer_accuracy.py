"""Answer accuracy for the generation benchmark.

The generation run reports lexical overlap scores. This script measures
something more direct for the lookup questions in evaluation/dataset.py:
whether each answer names the expected file and symbol, whether it refuses,
and whether the citations it references point into the expected file.

Run after a generation benchmark:

    python -m evaluation.answer_accuracy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from evaluation.dataset import create_default_dataset

REFUSAL = "not found in the provided code"
REFERENCE = re.compile(r"\[(\d+)\]")

METRICS = {
    "refused": "Refused (\"Not found in the provided code\")",
    "names_file": "Names the expected file (full path)",
    "names_file_lenient": "Names the expected file (file name)",
    "names_symbol": "Names the expected symbol",
    "has_reference": "References a citation, e.g. [1]",
    "cites_expected_file": "A referenced citation is in the expected file",
}


def _lookup(container: Any, query: str, index: int) -> Any:
    if isinstance(container, dict):
        return container.get(query, container.get(str(index)))
    if isinstance(container, list) and index < len(container):
        return container[index]
    return None


def _answer_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("answer", "")
    return str(value or "")


def _citation_files(value: Any) -> list[str]:
    files = []
    for item in value or []:
        if isinstance(item, dict):
            files.append(str(item.get("file_path") or item.get("file") or ""))
        else:
            files.append(str(item))
    return files


def names_file(answer: str, expected_files, lenient: bool = False) -> bool:
    text = answer.lower()
    for path in expected_files:
        if path.lower() in text:
            return True
        if lenient and PurePosixPath(path).name.lower() in text:
            return True
    return False


def names_symbol(answer: str, expected_symbols) -> bool:
    for symbol in expected_symbols:
        name = symbol.split(".")[-1]
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
        if re.search(pattern, answer):
            return True
    return False


def cited_files(answer: str, citation_files: list[str]) -> list[str]:
    files = []
    for match in REFERENCE.findall(answer):
        position = int(match) - 1
        if 0 <= position < len(citation_files):
            files.append(citation_files[position])
    return files


def score(example, answer: str, citation_files: list[str]) -> dict[str, Any]:
    expected_files = list(example.expected_files)
    referenced = cited_files(answer, citation_files)

    return {
        "query": example.query,
        "refused": REFUSAL in answer.lower(),
        "names_file": names_file(answer, expected_files) if expected_files else None,
        "names_file_lenient": (
            names_file(answer, expected_files, lenient=True) if expected_files else None
        ),
        "names_symbol": (
            names_symbol(answer, example.expected_symbols)
            if example.expected_symbols
            else None
        ),
        "has_reference": bool(referenced),
        "cites_expected_file": (
            any(path in expected_files for path in referenced) if expected_files else None
        ),
        "answer": answer,
    }


def _rate(results, key: str) -> dict[str, Any]:
    applicable = [r[key] for r in results if r[key] is not None]
    hits = sum(1 for value in applicable if value)
    return {
        "rate": round(hits / len(applicable), 4) if applicable else 0.0,
        "count": hits,
        "of": len(applicable),
    }


def summarize(results) -> dict[str, Any]:
    return {key: _rate(results, key) for key in METRICS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer accuracy for the generation benchmark")
    parser.add_argument(
        "--checkpoint",
        default="evaluation/results/benchmark-generation.checkpoint.json",
    )
    parser.add_argument("--output", default="evaluation/results/answer-accuracy.json")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
    answers = data.get("answers")
    citations = data.get("citations_by_query")

    results = []
    for index, example in enumerate(create_default_dataset()):
        answer = _answer_text(_lookup(answers, example.query, index))
        files = _citation_files(_lookup(citations, example.query, index))
        results.append(score(example, answer, files))

    found = sum(1 for r in results if r["answer"].strip())
    print(f"Answers found in checkpoint: {found}/{len(results)}")
    print()

    summary = summarize(results)
    for key, label in METRICS.items():
        s = summary[key]
        print(f"{label:<48} {s['rate']:.2f}  ({s['count']}/{s['of']})")

    print()
    print("Answers that do not name the expected file:")
    for r in results:
        if r["names_file_lenient"] is False:
            snippet = " ".join(r["answer"].split())[:90]
            print(f"  - {r['query']}")
            print(f"      {snippet}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "per_query": results}, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"Results saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

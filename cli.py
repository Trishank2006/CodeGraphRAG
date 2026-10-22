from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from application.service import CodeGraphRAGService


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=asdict, ensure_ascii=False))


def _print_index_summary(summary: object) -> None:
    data = asdict(summary)

    print("=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)
    print(f"Repository:   {data.get('repository')}")
    print(f"Files:        {data.get('files')}")
    print(f"Entities:     {data.get('entities')}")
    print(f"Chunks:       {data.get('chunks')}")
    print(f"Graph nodes:  {data.get('graph_nodes')}")
    print(f"Graph edges:  {data.get('graph_edges')}")
    print()


def _print_answer(summary: object, answer: object) -> None:
    print("=" * 60)
    print("CODEGRAPHRAG ANSWER")
    print("=" * 60)
    print()

    print(answer.answer)

    print()
    print("=" * 60)
    print("CITATIONS")
    print("=" * 60)

    if not answer.citations:
        print("No citations returned.")
        return

    for index, citation in enumerate(answer.citations, start=1):
        print()
        print(f"[{index}] {citation.file_path}")
        print(f"    Symbol: {citation.symbol}")
        print(f"    Lines:  {citation.start_line}-{citation.end_line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index and query code repositories"
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index = commands.add_parser(
        "index",
        help="Index a local repository",
    )
    index.add_argument("repository_path")
    index.add_argument("--name", dest="repository_name")

    ask = commands.add_parser(
        "ask",
        help="Index a local repository and generate a grounded answer",
    )
    ask.add_argument("repository_path")
    ask.add_argument("query")
    ask.add_argument("--name", dest="repository_name")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = CodeGraphRAGService()

    try:
        summary = service.index_repository(
            args.repository_path,
            args.repository_name,
        )

        if args.command == "index":
            _print_index_summary(summary)
            return 0

        answer = service.answer(args.query)

        _print_index_summary(summary)
        _print_answer(summary, answer)

        return 0

    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
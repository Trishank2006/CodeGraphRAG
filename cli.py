from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from application.service import (
    CodeGraphRAGService,
    IndexSummary,
    NoChunksGeneratedError,
    RepositoryNotIndexedError,
)
from generation.models import GeneratedAnswer


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=asdict, ensure_ascii=False))


def _print_index_summary(summary: IndexSummary) -> None:
    print("=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)
    print(f"Repository: {summary.repository}")
    print(f"Files: {summary.files}")
    print(f"Entities: {summary.entities}")
    print(f"Chunks: {summary.chunks}")
    print(f"Graph nodes: {summary.graph_nodes}")
    print(f"Graph edges: {summary.graph_edges}")
    print()


def _print_answer(answer: GeneratedAnswer) -> None:
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
        print(f" Symbol: {citation.symbol}")
        print(f" Lines: {citation.start_line}-{citation.end_line}")


def _add_exclude_argument(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Skip a directory or file relative to the repository root; "
            "repeatable, e.g. --exclude tests --exclude evaluation"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and query code repositories")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Index a local repository")
    index.add_argument("repository_path")
    index.add_argument("--name", dest="repository_name")
    _add_exclude_argument(index)

    ask = commands.add_parser(
        "ask",
        help="Answer a question about a repository, indexing it only if needed",
    )
    ask.add_argument("repository_path")
    ask.add_argument("query")
    ask.add_argument("--name", dest="repository_name")
    ask.add_argument(
        "--reindex",
        action="store_true",
        help="Re-index even if an index already exists (use after code changes)",
    )
    _add_exclude_argument(ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_name = (
        args.repository_name or Path(args.repository_path).resolve().name
    )
    service = CodeGraphRAGService()

    try:
        if args.command == "index":
            summary = service.index_repository(
                args.repository_path, repository_name, exclude=args.exclude
            )
            _print_index_summary(summary)
            return 0

        summary = None
        if args.reindex or not service.has_repository(repository_name):
            summary = service.index_repository(
                args.repository_path, repository_name, exclude=args.exclude
            )
        elif args.exclude:
            print(
                "Note: --exclude only applies when indexing; "
                "add --reindex to apply it.",
                file=sys.stderr,
            )

        answer = service.answer(args.query, repository=repository_name)

        if summary is not None:
            _print_index_summary(summary)
        _print_answer(answer)
        return 0

    except (NotADirectoryError, NoChunksGeneratedError, RepositoryNotIndexedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
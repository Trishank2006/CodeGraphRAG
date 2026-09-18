from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from application.service import CodeGraphRAGService


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=asdict))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and query code repositories")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Index a local repository")
    index.add_argument("repository_path")
    index.add_argument("--name", dest="repository_name")

    ask = commands.add_parser(
        "ask", help="Index a local repository and generate a grounded answer"
    )
    ask.add_argument("repository_path")
    ask.add_argument("query")
    ask.add_argument("--name", dest="repository_name")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = CodeGraphRAGService()

    try:
        summary = service.index_repository(args.repository_path, args.repository_name)
        if args.command == "index":
            _print_json(asdict(summary))
            return 0

        answer = service.answer(args.query)
        _print_json(
            {
                "index": asdict(summary),
                "answer": answer.answer,
                "citations": [asdict(citation) for citation in answer.citations],
            }
        )
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())

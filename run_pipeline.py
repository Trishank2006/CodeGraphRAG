"""Backward-compatible entry point for the production CodeGraphRAG CLI."""

from cli import main


if __name__ == "__main__":
    raise SystemExit(main())

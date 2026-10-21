
"""Opt-in test of the real Qdrant, Neo4j, embedding, and OpenAI integration.

Run only in an environment with Neo4j, Qdrant, and OPENAI_API_KEY configured:

    $env:CODEGRAPHRAG_RUN_LIVE_TESTS = "1"
    python -m pytest -v tests/test_live_end_to_end.py
"""

import os

import pytest

from application.service import CodeGraphRAGService


# Skip this module unless live integration tests are explicitly enabled
# AND the required OpenAI API key is available.
pytestmark = pytest.mark.skipif(
    os.getenv("CODEGRAPHRAG_RUN_LIVE_TESTS") != "1"
    or not os.getenv("OPENAI_API_KEY")
    or not os.getenv("OPENAI_BASE_URL"),
    reason=(
        "Live end-to-end test requires "
        "CODEGRAPHRAG_RUN_LIVE_TESTS=1, OPENAI_API_KEY, "
        "and OPENAI_BASE_URL"
    ),
)


def test_real_pipeline_indexes_and_answers_a_local_repository(tmp_path):
    source = tmp_path / "auth.py"
    source.write_text(
        "def authenticate_user(token):\n"
        "    return verify_token(token)\n",
        encoding="utf-8",
    )

    service = CodeGraphRAGService()

    try:
        summary = service.index_repository(
            str(tmp_path),
            "live-codegraphrag-test",
        )
        assert summary.chunks == 1

        answer = service.answer("How is authentication implemented?")
        assert answer.answer.strip()
        assert answer.citations

    finally:
        service.close()

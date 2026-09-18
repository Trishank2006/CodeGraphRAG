"""Opt-in test of the real Qdrant, Neo4j, embedding, and OpenAI integration.

Run only in an environment with Neo4j and OPENAI_API_KEY configured:
    CODEGRAPHRAG_RUN_LIVE_TESTS=1 pytest tests/test_live_end_to_end.py
"""

import os

import pytest

from application.service import CodeGraphRAGService


pytestmark = pytest.mark.skipif(
    os.getenv("CODEGRAPHRAG_RUN_LIVE_TESTS") != "1",
    reason="set CODEGRAPHRAG_RUN_LIVE_TESTS=1 to run external integration tests",
)


def test_real_pipeline_indexes_and_answers_a_local_repository(tmp_path):
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for real answer generation")

    source = tmp_path / "auth.py"
    source.write_text(
        "def authenticate_user(token):\n    return verify_token(token)\n",
        encoding="utf-8",
    )

    service = CodeGraphRAGService()
    try:
        summary = service.index_repository(str(tmp_path), "live-codegraphrag-test")
        assert summary.chunks == 1

        answer = service.answer("How is authentication implemented?")
        assert answer.answer.strip()
        assert answer.citations
    finally:
        service.close()

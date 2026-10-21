
from evaluation.config import EvaluationConfig


def test_create_uses_configured_llm_model(monkeypatch):
    monkeypatch.setenv("CODEGRAPHRAG_OPENAI_MODEL", "qwen2.5:3b")

    config = EvaluationConfig.create()

    assert config.llm_model == "qwen2.5:3b"


def test_create_uses_git_revision_when_available(monkeypatch):
    monkeypatch.setattr(
        "evaluation.config.get_repository_version",
        lambda: "339b541",
    )

    config = EvaluationConfig.create()

    assert config.repository_version == "339b541"

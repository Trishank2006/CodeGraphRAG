
import pytest

from retrieval.config import RetrievalConfig


def test_default_fusion_weight_is_disabled(monkeypatch):
    monkeypatch.delenv("CODEGRAPHRAG_FUSION_WEIGHT", raising=False)

    config = RetrievalConfig.from_env()

    assert config.fusion_weight is None


def test_fusion_weight_is_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("CODEGRAPHRAG_FUSION_WEIGHT", "0.5")

    config = RetrievalConfig.from_env()

    assert config.fusion_weight == 0.5


@pytest.mark.parametrize("weight", ["0", "0.25", "0.5", "0.75", "1"])
def test_valid_fusion_weights_are_accepted(monkeypatch, weight):
    monkeypatch.setenv("CODEGRAPHRAG_FUSION_WEIGHT", weight)

    config = RetrievalConfig.from_env()

    assert config.fusion_weight == float(weight)


@pytest.mark.parametrize("weight", ["-0.1", "1.1"])
def test_out_of_range_fusion_weight_is_rejected(monkeypatch, weight):
    monkeypatch.setenv("CODEGRAPHRAG_FUSION_WEIGHT", weight)

    with pytest.raises(ValueError, match="fusion_weight"):
        RetrievalConfig.from_env()


def test_non_numeric_fusion_weight_is_rejected(monkeypatch):
    monkeypatch.setenv("CODEGRAPHRAG_FUSION_WEIGHT", "invalid")

    with pytest.raises(ValueError, match="must be a number"):
        RetrievalConfig.from_env()

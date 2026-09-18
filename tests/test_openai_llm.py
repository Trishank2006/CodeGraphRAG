import pytest

from generation.llm import LLMError
from generation.openai_client import OpenAILLM


class FakeResponses:
    def create(self, **kwargs):
        assert kwargs["model"] == "test-model"
        assert kwargs["input"] == "Explain this code."
        return type("Response", (), {"output_text": "Grounded answer."})()


class FakeClient:
    responses = FakeResponses()


def test_openai_llm_uses_responses_api_client():
    llm = OpenAILLM(model="test-model", client=FakeClient())
    assert llm.generate("Explain this code.") == "Grounded answer."


def test_openai_llm_rejects_empty_prompt():
    llm = OpenAILLM(client=FakeClient())
    with pytest.raises(LLMError):
        llm.generate("   ")

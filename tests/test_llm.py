import pytest

from generation.llm import LLMClient, LLMError, MockLLM, generate_text


def test_mock_llm_is_an_llm_client():
    client = MockLLM("test response")

    assert isinstance(client, LLMClient)


def test_mock_llm_returns_response():
    client = MockLLM("Hello from the model.")

    result = client.generate("Explain authentication.")

    assert result == "Hello from the model."


def test_mock_llm_records_prompt():
    client = MockLLM("response")

    client.generate("first prompt")
    client.generate("second prompt")

    assert client.received_prompts == [
        "first prompt",
        "second prompt",
    ]


def test_generate_text_uses_client():
    client = MockLLM("generated answer")

    result = generate_text(
        client,
        "test prompt",
    )

    assert result == "generated answer"


def test_generate_text_rejects_empty_prompt():
    client = MockLLM("response")

    with pytest.raises(LLMError):
        generate_text(client, "   ")


def test_mock_llm_rejects_empty_prompt():
    client = MockLLM("response")

    with pytest.raises(LLMError):
        client.generate("")


def test_generate_text_rejects_empty_response():
    client = MockLLM("   ")

    with pytest.raises(LLMError):
        generate_text(client, "test prompt")


def test_generate_text_rejects_non_string_response():
    class BadLLM(LLMClient):
        def generate(self, prompt: str):
            return 123

    with pytest.raises(LLMError):
        generate_text(
            BadLLM(),
            "test prompt",
        )


def test_generate_text_wraps_provider_errors():
    class FailingLLM(LLMClient):
        def generate(self, prompt: str) -> str:
            raise RuntimeError("provider failure")

    with pytest.raises(LLMError):
        generate_text(
            FailingLLM(),
            "test prompt",
        )


def test_llm_error_is_runtime_error():
    assert issubclass(LLMError, RuntimeError)
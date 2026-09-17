from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Raised when LLM generation fails."""


class LLMClient(ABC):
    """
    Provider-independent interface for language model generation.

    The rest of CodeGraphRAG should depend on this interface rather than
    directly depending on a specific model SDK or provider.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Implementations should return the generated text as a string.
        """
        raise NotImplementedError


class MockLLM(LLMClient):
    """
    Deterministic LLM implementation for tests.

    This avoids downloading or connecting to a real model during tests.
    """

    def __init__(self, response: str) -> None:
        self.response = response
        self.received_prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise LLMError("Prompt must not be empty")

        self.received_prompts.append(prompt)
        return self.response


def generate_text(
    client: LLMClient,
    prompt: str,
) -> str:
    """
    Generate text through an LLMClient with basic validation.
    """
    if not prompt.strip():
        raise LLMError("Prompt must not be empty")

    try:
        response = client.generate(prompt)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError("LLM generation failed") from exc

    if not isinstance(response, str):
        raise LLMError("LLM client must return a string")

    if not response.strip():
        raise LLMError("LLM returned an empty response")

    return response
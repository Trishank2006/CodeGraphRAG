from __future__ import annotations

import os
from typing import Any

from generation.llm import LLMClient, LLMError


class OpenAILLM(LLMClient):
    """Production ``LLMClient`` backed by the OpenAI Responses API.

    The OpenAI SDK is imported only when the client is constructed so unit
    tests and local indexing do not require an API key or the SDK.  By
    default, the SDK reads ``OPENAI_API_KEY`` from the environment.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model

        if client is not None:
            self.client = client
            return

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise LLMError(
                "OPENAI_API_KEY must be set to use the OpenAI LLM provider"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The OpenAI SDK is not installed; install requirements.txt"
            ) from exc

        self.client = OpenAI(api_key=resolved_key)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise LLMError("Prompt must not be empty")

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            output = response.output_text
        except Exception as exc:
            raise LLMError("OpenAI generation failed") from exc

        if not isinstance(output, str) or not output.strip():
            raise LLMError("OpenAI returned an empty response")

        return output

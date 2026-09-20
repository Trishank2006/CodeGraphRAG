from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Load local .env variables into os.environ
load_dotenv(override=True)

from generation.llm import LLMClient, LLMError


class OpenAILLM(LLMClient):
    """Production ``LLMClient`` backed by OpenAI or an OpenAI-compatible API (e.g. Grok / xAI).

    The SDK reads ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and ``OPENAI_MODEL``
    directly from the environment or local .env file.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "grok-2-latest")

        if client is not None:
            self.client = client
            return

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise LLMError(
                "OPENAI_API_KEY must be set to use the OpenAI/Grok LLM provider"
            )

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The OpenAI SDK is not installed; install requirements.txt"
            ) from exc

        if resolved_base_url:
            self.client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)
        else:
            self.client = OpenAI(api_key=resolved_key)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise LLMError("Prompt must not be empty")

        try:
            # 1. Standard OpenAI and Grok (xAI) Chat Completions API
            if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                output = response.choices[0].message.content
            # 2. Fallback for legacy / specific OpenAI Responses endpoint
            elif hasattr(self.client, "responses"):
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                )
                output = getattr(response, "output_text", None) or str(response)
            else:
                raise LLMError("Unsupported OpenAI client interface")
        except Exception as exc:
            raise LLMError(f"LLM generation failed: {exc}") from exc

        if not isinstance(output, str) or not output.strip():
            raise LLMError("LLM returned an empty response")

        return output
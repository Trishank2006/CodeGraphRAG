
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Load local .env variables into os.environ.
load_dotenv(override=True)

from generation.llm import LLMClient, LLMError


class OpenAILLM(LLMClient):
    """LLM client for OpenAI-compatible APIs, including local Ollama."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = (
            model
            or os.getenv("OPENAI_MODEL")
            or "grok-2-latest"
        )

        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or ""
        )

        self.is_local_ollama = (
            "localhost:11434" in self.base_url
            or "127.0.0.1:11434" in self.base_url
        )

        if client is not None:
            self.client = client
            return

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise LLMError(
                "OPENAI_API_KEY must be set to use the LLM provider"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The OpenAI SDK is not installed; install requirements.txt"
            ) from exc

        if self.base_url:
            self.client = OpenAI(
                api_key=resolved_key,
                base_url=self.base_url,
            )
        else:
            self.client = OpenAI(api_key=resolved_key)

    def generate(self, prompt: str) -> str:
        """Generate a response with reproducible sampling settings."""
        if not prompt.strip():
            raise LLMError("Prompt must not be empty")

        try:
            # OpenAI-compatible Chat Completions API.
            if (
                hasattr(self.client, "chat")
                and hasattr(self.client.chat, "completions")
            ):
                request_args: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "seed": 42,
                }

                # Ollama-specific context-window configuration.
                if self.is_local_ollama:
                    request_args["extra_body"] = {
                        "options": {
                            "num_ctx": 8192,
                            "seed": 42,
                            "temperature": 0,
                        }
                    }

                response = self.client.chat.completions.create(
                    **request_args
                )
                output = response.choices[0].message.content

            # Fallback for clients exposing the Responses API.
            elif hasattr(self.client, "responses"):
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                )
                output = (
                    getattr(response, "output_text", None)
                    or str(response)
                )

            else:
                raise LLMError(
                    "Unsupported OpenAI client interface"
                )

        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"LLM generation failed: {exc}"
            ) from exc

        if not isinstance(output, str) or not output.strip():
            raise LLMError("LLM returned an empty response")

        return output

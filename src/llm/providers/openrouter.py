"""OpenRouter LLM provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.llm.providers.base import (
    ChatMessage,
    LLMChunk,
    LLMResponse,
    TokenUsage,
)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    """LLM provider that calls the OpenRouter API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request and return the full response."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            **kwargs,
        }
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        usage_data = data.get("usage")

        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=TokenUsage(**usage_data) if usage_data else None,
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Send a streaming chat completion request and yield chunks."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            **kwargs,
        }
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str.strip() == "[DONE]":
                    break
                chunk_data = json.loads(data_str)
                delta = chunk_data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if not content:
                    continue
                finish_reason = chunk_data["choices"][0].get("finish_reason")
                yield LLMChunk(content=content, finish_reason=finish_reason)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

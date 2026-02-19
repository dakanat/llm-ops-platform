"""Anthropic LLM provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.llm.providers.base import (
    ChatMessage,
    LLMChunk,
    LLMResponse,
    Role,
    TokenUsage,
)

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider:
    """LLM provider that calls the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    @staticmethod
    def _extract_system(
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[ChatMessage]]:
        """Separate system messages from the message list.

        Returns the joined system prompt (or None) and the remaining messages.
        """
        system_parts: list[str] = []
        non_system: list[ChatMessage] = []
        for msg in messages:
            if msg.role == Role.system:
                system_parts.append(msg.content)
            else:
                non_system.append(msg)
        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, non_system

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request and return the full response."""
        system_text, non_system_messages = self._extract_system(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in non_system_messages],
            "max_tokens": kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS),
            **kwargs,
        }
        if system_text is not None:
            payload["system"] = system_text

        response = await self._client.post("/v1/messages", json=payload)
        response.raise_for_status()

        data = response.json()
        content = data["content"][0]["text"]
        usage_data = data.get("usage")
        usage = None
        if usage_data:
            input_tokens = usage_data["input_tokens"]
            output_tokens = usage_data["output_tokens"]
            usage = TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

        return LLMResponse(
            content=content,
            model=data["model"],
            usage=usage,
            finish_reason=data.get("stop_reason"),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMChunk, None]:
        """Send a streaming chat completion request and yield chunks."""
        system_text, non_system_messages = self._extract_system(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in non_system_messages],
            "max_tokens": kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS),
            "stream": True,
            **kwargs,
        }
        if system_text is not None:
            payload["system"] = system_text

        async with self._client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            current_event: str | None = None
            async for line in response.aiter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("event: "):
                    current_event = line[len("event: ") :]
                    continue
                if not line.startswith("data: "):
                    continue
                if current_event == "message_stop":
                    break
                if current_event != "content_block_delta":
                    continue
                data_str = line[len("data: ") :]
                chunk_data = json.loads(data_str)
                delta = chunk_data.get("delta", {})
                if delta.get("type") != "text_delta":
                    continue
                text = delta.get("text", "")
                if text:
                    yield LLMChunk(content=text)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

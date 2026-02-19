"""Gemini LLM provider implementation (Google OpenAI-compatible endpoint)."""

from __future__ import annotations

import asyncio
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

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

_MAX_RETRIES = 5
_BASE_DELAY = 2.0  # seconds (conservative for free tier)
_MAX_DELAY = 60.0  # seconds
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _calc_delay(response: httpx.Response, attempt: int) -> float:
    """Calculate retry delay from Retry-After header or exponential backoff."""
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    delay: float = _BASE_DELAY * (2**attempt)
    return min(delay, _MAX_DELAY)


def _build_status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    """Build an HTTPStatusError from a response with a retryable status code."""
    return httpx.HTTPStatusError(
        f"{response.status_code} Error",
        request=response.request if hasattr(response, "request") else httpx.Request("POST", ""),
        response=response,
    )


class GeminiProvider:
    """LLM provider that calls the Gemini API (OpenAI-compatible endpoint)."""

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
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            response = await self._client.post("/chat/completions", json=payload)
            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_exc = _build_status_error(response)
                if attempt < _MAX_RETRIES:
                    delay = _calc_delay(response, attempt)
                    await asyncio.sleep(delay)
                    continue
                raise last_exc
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

        # Unreachable in practice, but satisfies the type checker.
        assert last_exc is not None  # noqa: S101
        raise last_exc

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
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    last_exc = _build_status_error(response)
                    if attempt < _MAX_RETRIES:
                        delay = _calc_delay(response, attempt)
                        await asyncio.sleep(delay)
                        continue
                    raise last_exc
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
                return

        # Unreachable in practice, but satisfies the type checker.
        assert last_exc is not None  # noqa: S101
        raise last_exc

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

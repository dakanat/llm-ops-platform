"""Shared fixtures for unit tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from unittest.mock import AsyncMock

import pytest
from src.llm.providers.base import (
    LLMChunk,
    LLMResponse,
    TokenUsage,
)


@pytest.fixture
def make_mock_llm_provider() -> Callable[..., AsyncMock]:
    """Factory for mock LLM providers with complete + stream support."""

    def _factory(
        response: LLMResponse | None = None,
        chunks: list[LLMChunk] | None = None,
    ) -> AsyncMock:
        provider = AsyncMock()
        provider.complete.return_value = response or LLMResponse(
            content="Hello!",
            model="test-model",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            finish_reason="stop",
        )

        async def _stream(*args: object, **kwargs: object) -> AsyncGenerator[LLMChunk, None]:
            for chunk in chunks or [LLMChunk(content="Hello!", finish_reason="stop")]:
                yield chunk

        provider.stream = _stream
        return provider

    return _factory


@pytest.fixture
def make_llm_response() -> Callable[..., LLMResponse]:
    """Factory for LLMResponse instances."""

    def _factory(
        content: str = "回答テキスト",
        model: str = "test-model",
        usage: TokenUsage | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
        )

    return _factory

"""LLM provider Protocol and shared data models."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class Role(StrEnum):
    """Chat message roles."""

    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Token usage statistics from an LLM response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    """Complete (non-streaming) response from an LLM provider."""

    content: str
    model: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None


class LLMChunk(BaseModel):
    """A single chunk from a streaming LLM response."""

    content: str
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must satisfy."""

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMChunk, None]: ...

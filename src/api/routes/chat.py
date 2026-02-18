"""Chat endpoint with streaming and non-streaming support."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.dependencies import get_llm_model, get_llm_provider
from src.llm.providers.base import ChatMessage, LLMProvider, TokenUsage

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request body."""

    messages: list[ChatMessage]
    stream: bool = False


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    content: str
    model: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None


@router.post("/chat", response_model=None)
async def chat(
    request: ChatRequest,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    model: Annotated[str, Depends(get_llm_model)],
) -> ChatResponse | StreamingResponse:
    """Chat completion endpoint supporting streaming and non-streaming modes."""
    if request.stream:
        return _stream_response(provider, model, request.messages)

    try:
        response = await provider.complete(messages=request.messages, model=model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ChatResponse(
        content=response.content,
        model=model,
        usage=response.usage,
        finish_reason=response.finish_reason,
    )


def _stream_response(
    provider: LLMProvider,
    model: str,
    messages: list[ChatMessage],
) -> StreamingResponse:
    """Build an SSE streaming response."""

    async def _generate() -> AsyncGenerator[str, None]:
        async for chunk in provider.stream(messages=messages, model=model):
            data = json.dumps({"content": chunk.content, "finish_reason": chunk.finish_reason})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
    )

"""Web chat interface routes."""

from __future__ import annotations

import html
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from src.api.dependencies import get_llm_model, get_llm_provider
from src.llm.providers.base import ChatMessage, LLMProvider, Role
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: CurrentWebUser) -> Response:
    """Display the chat page."""
    context = {"user": user, "active_page": "chat"}
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "chat/page.html", context)
    return templates.TemplateResponse(request, "chat/page.html", context)


@router.post("/chat/send", response_class=HTMLResponse)
async def chat_send(
    request: Request,
    user: CurrentWebUser,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    model: Annotated[str, Depends(get_llm_model)],
) -> Response:
    """Send a message and return the response as an HTML fragment."""
    form = await request.form()
    message = str(form.get("message", "")).strip()

    if not message:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Message cannot be empty"},
        )

    messages = [ChatMessage(role=Role.user, content=message)]

    try:
        response = await provider.complete(messages=messages, model=model)
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"LLM error: {e}"},
        )

    return templates.TemplateResponse(
        request,
        "chat/message.html",
        {
            "user_message": message,
            "assistant_message": response.content,
            "model": model,
        },
    )


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    user: CurrentWebUser,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    model: Annotated[str, Depends(get_llm_model)],
) -> StreamingResponse:
    """Stream the LLM response as SSE events with HTML fragments."""
    form = await request.form()
    message = str(form.get("message", "")).strip()

    messages = [ChatMessage(role=Role.user, content=message)]

    async def _generate() -> AsyncGenerator[str, None]:
        # Send user message first
        escaped_msg = html.escape(message)
        yield (
            f"event: message\n"
            f'data: <div class="chat chat-end"><div class="chat-bubble chat-bubble-primary">'
            f"{escaped_msg}</div></div>\n\n"
        )

        # Stream assistant response
        yield (
            "event: message\n"
            'data: <div class="chat chat-start"><div class="chat-bubble" id="stream-target">\n\n'
        )

        async for chunk in provider.stream(messages=messages, model=model):
            escaped = html.escape(chunk.content)
            yield f"event: message\ndata: {escaped}\n\n"

        yield ("event: message\ndata: </div></div>\n\n")

        yield "event: done\ndata: \n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

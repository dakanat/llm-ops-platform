"""Web chat interface routes (Agent-integrated)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse

from src.agent import AgentError
from src.agent.runtime import AgentResult, AgentRuntime
from src.agent.tools.registry import ToolRegistry
from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry
from src.llm.providers.base import LLMProvider
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


def _create_runtime(
    provider: LLMProvider,
    model: str,
    registry: ToolRegistry,
) -> AgentRuntime:
    """Create an AgentRuntime instance. Extracted for testability."""
    return AgentRuntime(
        llm_provider=provider,
        model=model,
        tool_registry=registry,
    )


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
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> Response:
    """Send a message via Agent and return the response as an HTML fragment."""
    form = await request.form()
    message = str(form.get("message", "")).strip()

    if not message:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Message cannot be empty"},
        )

    runtime = _create_runtime(provider, model, registry)

    try:
        result: AgentResult = await runtime.run(message)
    except AgentError as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"Agent error: {e}"},
        )

    # Collect RAG sources from steps
    sources: list[dict[str, object]] = []
    for step in result.steps:
        if step.metadata and "sources" in step.metadata:
            sources.extend(step.metadata["sources"])

    return templates.TemplateResponse(
        request,
        "chat/message.html",
        {
            "user_message": message,
            "assistant_message": result.answer,
            "model": model,
            "steps": result.steps,
            "total_steps": result.total_steps,
            "stopped_by_max_steps": result.stopped_by_max_steps,
            "sources": sources,
        },
    )

"""Web Agent interface routes."""

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


@router.get("/agent", response_class=HTMLResponse)
async def agent_page(request: Request, user: CurrentWebUser) -> Response:
    """Display the Agent page."""
    return templates.TemplateResponse(
        request, "agent/page.html", {"user": user, "active_page": "agent"}
    )


@router.post("/agent/run", response_class=HTMLResponse)
async def agent_run(
    request: Request,
    user: CurrentWebUser,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    model: Annotated[str, Depends(get_llm_model)],
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> Response:
    """Execute an agent query and return the result as HTML."""
    form = await request.form()
    query = str(form.get("query", "")).strip()

    if not query:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Query cannot be empty"},
        )

    runtime = _create_runtime(provider, model, registry)

    try:
        result: AgentResult = await runtime.run(query)
    except AgentError as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"Agent error: {e}"},
        )

    return templates.TemplateResponse(
        request,
        "agent/result.html",
        {
            "query": query,
            "answer": result.answer,
            "steps": result.steps,
            "total_steps": result.total_steps,
            "stopped_by_max_steps": result.stopped_by_max_steps,
        },
    )

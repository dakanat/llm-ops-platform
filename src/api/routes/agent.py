"""Agent execution endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.agent import AgentError
from src.agent.runtime import AgentRuntime
from src.agent.tools.registry import ToolRegistry
from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry
from src.llm.providers.base import LLMProvider
from src.security.permission import Permission, require_permission

router = APIRouter(prefix="/agent")


class AgentStepResponse(BaseModel):
    """Agent の1ステップのレスポンス。"""

    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    is_error: bool = False


class AgentRunRequest(BaseModel):
    """Agent 実行リクエスト。"""

    query: str = Field(min_length=1)
    max_steps: int = 10


class AgentRunResponse(BaseModel):
    """Agent 実行レスポンス。"""

    answer: str
    steps: list[AgentStepResponse]
    total_steps: int
    stopped_by_max_steps: bool


@router.post("/run")
async def agent_run(
    request: AgentRunRequest,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    model: Annotated[str, Depends(get_llm_model)],
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    _user: Annotated[None, Depends(require_permission(Permission.AGENT_RUN))],
) -> AgentRunResponse:
    """Execute an agent with the given query."""
    runtime = AgentRuntime(
        llm_provider=provider,
        model=model,
        tool_registry=registry,
        max_steps=request.max_steps,
    )

    try:
        result = await runtime.run(request.query)
    except AgentError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return AgentRunResponse(
        answer=result.answer,
        steps=[
            AgentStepResponse(
                thought=step.thought,
                action=step.action,
                action_input=step.action_input,
                observation=step.observation,
                is_error=step.is_error,
            )
            for step in result.steps
        ],
        total_steps=result.total_steps,
        stopped_by_max_steps=result.stopped_by_max_steps,
    )

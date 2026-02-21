"""Web chat interface routes (Agent-integrated)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.responses import StreamingResponse

from src.agent import AgentError
from src.agent.runtime import AgentResult, AgentRuntime
from src.agent.tools.registry import ToolRegistry
from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry
from src.db.session import get_session
from src.llm.providers.base import ChatMessage, LLMProvider, Role
from src.web.dependencies import CurrentWebUser
from src.web.services.conversation import ConversationService
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


def _create_conversation_service(session: AsyncSession) -> ConversationService:
    """Create a ConversationService instance. Extracted for testability."""
    return ConversationService(session=session)


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    user: CurrentWebUser,
    session: Annotated[AsyncSession, Depends(get_session)],
    cid: str | None = None,
) -> Response:
    """Display the chat page, optionally loading a conversation."""
    context: dict[str, object] = {
        "user": user,
        "active_page": "chat",
        "conversation_id": cid or "",
        "messages": [],
    }

    if cid:
        try:
            conv_id = uuid.UUID(cid)
        except ValueError:
            pass
        else:
            service = _create_conversation_service(session)
            conv = await service.get_conversation(conv_id, uuid.UUID(user.sub))
            if conv:
                messages = await service.get_messages(conv_id)
                context["messages"] = messages
                context["conversation_id"] = str(conv.id)

    return templates.TemplateResponse(request, "chat/page.html", context)


@router.get("/chat/conversations", response_class=HTMLResponse)
async def list_conversations(
    request: Request,
    user: CurrentWebUser,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Return conversation list HTML fragment for the sidebar."""
    service = _create_conversation_service(session)
    conversations = await service.list_conversations(uuid.UUID(user.sub))
    return templates.TemplateResponse(
        request,
        "chat/sidebar.html",
        {"conversations": conversations},
    )


@router.get("/chat/conversations/{conversation_id}", response_class=HTMLResponse)
async def get_conversation(
    request: Request,
    user: CurrentWebUser,
    conversation_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Return conversation messages as HTML fragment."""
    service = _create_conversation_service(session)
    conv = await service.get_conversation(conversation_id, uuid.UUID(user.sub))
    if not conv:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Conversation not found"},
        )
    messages = await service.get_messages(conversation_id)
    return templates.TemplateResponse(
        request,
        "chat/conversation_messages.html",
        {"messages": messages},
    )


@router.post("/chat/conversations/new", response_class=HTMLResponse)
async def create_conversation(
    request: Request,
    user: CurrentWebUser,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Create a new conversation and redirect to it."""
    service = _create_conversation_service(session)
    conv = await service.create_conversation(uuid.UUID(user.sub))
    response = Response(status_code=200)
    response.headers["HX-Redirect"] = f"/web/chat?cid={conv.id}"
    return response


@router.delete("/chat/conversations/{conversation_id}", response_class=HTMLResponse)
async def delete_conversation(
    request: Request,
    user: CurrentWebUser,
    conversation_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Delete a conversation and refresh the sidebar."""
    service = _create_conversation_service(session)
    await service.delete_conversation(conversation_id, uuid.UUID(user.sub))
    # Return refreshed sidebar
    conversations = await service.list_conversations(uuid.UUID(user.sub))
    return templates.TemplateResponse(
        request,
        "chat/sidebar.html",
        {"conversations": conversations},
    )


@router.post("/chat/send", response_class=HTMLResponse)
async def chat_send(
    request: Request,
    user: CurrentWebUser,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    model: Annotated[str, Depends(get_llm_model)],
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    session: Annotated[AsyncSession, Depends(get_session)],
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

    service = _create_conversation_service(session)
    conversation_id_str = str(form.get("conversation_id", "")).strip()
    conv_id: uuid.UUID | None = None
    user_id = uuid.UUID(user.sub)

    # Resolve or create conversation
    if conversation_id_str:
        try:
            conv_id = uuid.UUID(conversation_id_str)
        except ValueError:
            conv_id = None

    if conv_id:
        conv = await service.get_conversation(conv_id, user_id)
        if not conv:
            conv_id = None

    if not conv_id:
        conv = await service.create_conversation(user_id, title=message[:50])
        conv_id = conv.id

    # Load conversation history
    db_messages = await service.get_messages(conv_id)
    conversation_history: list[ChatMessage] = []
    for msg in db_messages:
        if msg.role in ("user", "assistant"):
            role = Role.user if msg.role == "user" else Role.assistant
            conversation_history.append(ChatMessage(role=role, content=msg.content))

    runtime = _create_runtime(provider, model, registry)

    try:
        result: AgentResult = await runtime.run(message, conversation_history=conversation_history)
    except AgentError as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"Agent error: {e}"},
        )

    # Persist messages
    await service.add_message(conv_id, "user", message)
    await service.add_message(conv_id, "assistant", result.answer)

    # Set title for new conversations
    if not conversation_id_str:
        await service.update_title(conv_id, message[:50])

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
            "conversation_id": str(conv_id),
        },
    )


def _render_template(template_name: str, context: dict[str, object]) -> str:
    """Render a Jinja2 template to string."""
    tmpl = templates.get_template(template_name)
    return tmpl.render(context)


def _sse_event(event: str, data: str) -> str:
    """Format a single SSE event."""
    escaped = json.dumps(data)
    return f"event: {event}\ndata: {escaped}\n\n"


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    user: CurrentWebUser,
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    model: Annotated[str, Depends(get_llm_model)],
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    session: Annotated[AsyncSession, Depends(get_session)],
    message: str = "",
    conversation_id: str = "",
) -> StreamingResponse:
    """SSE streaming endpoint for chat messages."""
    from src.agent.runtime import AgentAnswerEvent as _AnswerEvent
    from src.agent.runtime import AgentStepEvent as _StepEvent

    message = message.strip()

    async def _generate() -> AsyncGenerator[str, None]:
        if not message:
            yield _sse_event("error", json.dumps({"error": "Message cannot be empty"}))
            yield "event: done\ndata: {}\n\n"
            return

        service = _create_conversation_service(session)
        conv_id: uuid.UUID | None = None
        user_id = uuid.UUID(user.sub)

        if conversation_id:
            try:
                conv_id = uuid.UUID(conversation_id)
            except ValueError:
                conv_id = None

        if conv_id:
            conv = await service.get_conversation(conv_id, user_id)
            if not conv:
                conv_id = None

        if not conv_id:
            conv = await service.create_conversation(user_id, title=message[:50])
            conv_id = conv.id

        # Load history
        db_messages = await service.get_messages(conv_id)
        history: list[ChatMessage] = []
        for msg in db_messages:
            if msg.role in ("user", "assistant"):
                role = Role.user if msg.role == "user" else Role.assistant
                history.append(ChatMessage(role=role, content=msg.content))

        # Send user message bubble
        user_html = _render_template("chat/stream_user.html", {"user_message": message})
        yield _sse_event("user-message", user_html)

        # Send conversation ID
        yield _sse_event("conversation-id", json.dumps({"id": str(conv_id)}))

        runtime = _create_runtime(provider, model, registry)
        answer_text = ""

        try:
            async for event in runtime.run_streaming(message, conversation_history=history):
                if isinstance(event, _StepEvent):
                    step_html = _render_template(
                        "chat/stream_step.html",
                        {"step": event.step},
                    )
                    yield _sse_event("agent-step", step_html)

                elif isinstance(event, _AnswerEvent):
                    answer_text = event.answer
                    answer_html = _render_template(
                        "chat/stream_answer.html",
                        {
                            "answer": event.answer,
                            "model": model,
                            "total_steps": event.total_steps,
                            "stopped_by_max_steps": event.stopped_by_max_steps,
                            "sources": event.sources,
                        },
                    )
                    yield _sse_event("agent-answer", answer_html)
        except AgentError as e:
            yield _sse_event("error", json.dumps({"error": str(e)}))

        # Persist messages
        await service.add_message(conv_id, "user", message)
        if answer_text:
            await service.add_message(conv_id, "assistant", answer_text)

        if not conversation_id:
            await service.update_title(conv_id, message[:50])

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

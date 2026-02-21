"""Tests for web chat routes (Agent-integrated)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.agent.runtime import AgentResult
from src.agent.state import AgentStep
from src.config import Settings
from src.main import create_app

from tests.unit.web.conftest import AuthCookies


@pytest.fixture(scope="module")
def test_app() -> FastAPI:
    """Create a FastAPI app with rate limiting disabled."""
    settings = Settings(rate_limit_enabled=False)
    return create_app(settings)


@pytest_asyncio.fixture(scope="module")
async def client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Module-scoped async HTTP test client."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
        follow_redirects=False,
    ) as c:
        yield c


@pytest.fixture
def admin_token() -> str:
    """Return a valid JWT token."""
    from uuid import UUID

    from src.api.middleware.auth import create_access_token

    return create_access_token(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        email="admin@example.com",
        role="admin",
    )


class TestChatPage:
    """Tests for GET /web/chat."""

    async def test_unauthenticated_redirects_to_login(self, client: AsyncClient) -> None:
        resp = await client.get("/web/chat")
        assert resp.status_code == 303
        assert "/web/login" in resp.headers["location"]

    async def test_authenticated_returns_chat_page(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Chat" in resp.text

    async def test_send_button_has_id(self, client: AsyncClient, admin_token: str) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat")
        assert 'id="send-btn"' in resp.text

    async def test_form_does_not_use_global_loading_indicator(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat")
        assert 'hx-indicator="#loading-indicator"' not in resp.text

    async def test_cancel_button_exists(self, client: AsyncClient, admin_token: str) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat")
        assert 'id="cancel-btn"' in resp.text


def _mock_conversation_service() -> AsyncMock:
    """Create a mock ConversationService for chat_send tests."""
    import uuid

    from src.db.models import Conversation

    mock = AsyncMock()
    conv = Conversation(
        id=uuid.UUID("00000000-0000-0000-0000-000000000050"),
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
    )
    mock.get_conversation.return_value = None  # No existing conversation
    mock.create_conversation.return_value = conv
    mock.get_messages.return_value = []
    mock.add_message.return_value = None
    mock.update_title.return_value = None
    return mock


class TestChatSend:
    """Tests for POST /web/chat/send (Agent-integrated)."""

    @pytest.fixture(autouse=True)
    def _patch_conversation_service(self) -> Generator[None, None, None]:
        """Patch _create_conversation_service for all tests in this class."""
        import src.web.routes.chat as chat_module

        original = chat_module._create_conversation_service
        chat_module._create_conversation_service = lambda *a, **kw: _mock_conversation_service()
        yield
        chat_module._create_conversation_service = original

    async def test_send_returns_agent_response(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        mock_runtime = AsyncMock()
        mock_runtime.run.return_value = AgentResult(
            answer="Hello from the Agent!",
            steps=[AgentStep(thought="I know the answer.")],
            total_steps=1,
            stopped_by_max_steps=False,
        )

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        import src.web.routes.chat as chat_module

        original_create = chat_module._create_runtime

        def _mock_create(*args: object, **kwargs: object) -> AsyncMock:
            return mock_runtime

        chat_module._create_runtime = _mock_create

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/chat/send",
                    data={"message": "Hello"},
                )
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
            assert "Hello from the Agent!" in resp.text
        finally:
            chat_module._create_runtime = original_create
            test_app.dependency_overrides.clear()

    async def test_send_empty_message_returns_error(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/chat/send",
                data={"message": ""},
            )
        assert resp.status_code == 200
        body = resp.text
        assert "alert" in body.lower() or "error" in body.lower()

    async def test_send_with_tool_steps(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        mock_runtime = AsyncMock()
        mock_runtime.run.return_value = AgentResult(
            answer="The answer is 42.",
            steps=[
                AgentStep(
                    thought="I need to calculate this.",
                    action="calculator",
                    action_input="6 * 7",
                    observation="42",
                ),
                AgentStep(thought="Now I know the answer."),
            ],
            total_steps=2,
            stopped_by_max_steps=False,
        )

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        import src.web.routes.chat as chat_module

        original_create = chat_module._create_runtime

        def _mock_create(*args: object, **kwargs: object) -> AsyncMock:
            return mock_runtime

        chat_module._create_runtime = _mock_create

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/chat/send",
                    data={"message": "What is 6 * 7?"},
                )
            assert resp.status_code == 200
            assert "42" in resp.text
            assert "calculator" in resp.text
        finally:
            chat_module._create_runtime = original_create
            test_app.dependency_overrides.clear()

    async def test_send_with_rag_sources(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        mock_runtime = AsyncMock()
        mock_runtime.run.return_value = AgentResult(
            answer="Paris is the capital.",
            steps=[
                AgentStep(
                    thought="Searching for information.",
                    action="search",
                    action_input="capital of France",
                    observation="Paris is the capital of France.",
                    metadata={
                        "sources": [
                            {
                                "document_id": "doc-123",
                                "chunk_index": 0,
                                "content": "France's capital is Paris.",
                            }
                        ]
                    },
                ),
                AgentStep(thought="I have the answer."),
            ],
            total_steps=2,
            stopped_by_max_steps=False,
        )

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        import src.web.routes.chat as chat_module

        original_create = chat_module._create_runtime

        def _mock_create(*args: object, **kwargs: object) -> AsyncMock:
            return mock_runtime

        chat_module._create_runtime = _mock_create

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/chat/send",
                    data={"message": "What is the capital of France?"},
                )
            assert resp.status_code == 200
            assert "Paris is the capital." in resp.text
            assert "doc-123" in resp.text
        finally:
            chat_module._create_runtime = original_create
            test_app.dependency_overrides.clear()

    async def test_sources_display_uses_collapse_structure(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        mock_runtime = AsyncMock()
        mock_runtime.run.return_value = AgentResult(
            answer="Paris is the capital.",
            steps=[
                AgentStep(
                    thought="Searching.",
                    action="search",
                    action_input="capital of France",
                    observation="Paris is the capital of France.",
                    metadata={
                        "sources": [
                            {
                                "document_id": "doc-abc-123",
                                "chunk_index": 0,
                                "content": "France's capital is Paris.",
                            },
                            {
                                "document_id": "doc-def-456",
                                "chunk_index": 1,
                                "content": "Paris has the Eiffel Tower.",
                            },
                        ]
                    },
                ),
                AgentStep(thought="I have the answer."),
            ],
            total_steps=2,
            stopped_by_max_steps=False,
        )

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        import src.web.routes.chat as chat_module

        original_create = chat_module._create_runtime

        def _mock_create(*args: object, **kwargs: object) -> AsyncMock:
            return mock_runtime

        chat_module._create_runtime = _mock_create

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/chat/send",
                    data={"message": "What is the capital of France?"},
                )
            assert resp.status_code == 200
            body = resp.text
            # Outer collapse with source count
            assert "Sources (2)" in body
            assert 'class="collapse' in body
            # Inner source collapses with chunk badges
            assert "Chunk #0" in body
            assert "Chunk #1" in body
            # Document ID truncated to first 8 chars
            assert "doc-abc-..." in body
        finally:
            chat_module._create_runtime = original_create
            test_app.dependency_overrides.clear()

    async def test_agent_error_returns_error_toast(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        from src.agent import AgentError

        mock_runtime = AsyncMock()
        mock_runtime.run.side_effect = AgentError("LLM call failed")

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        import src.web.routes.chat as chat_module

        original_create = chat_module._create_runtime

        def _mock_create(*args: object, **kwargs: object) -> AsyncMock:
            return mock_runtime

        chat_module._create_runtime = _mock_create

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/chat/send",
                    data={"message": "Hello"},
                )
            assert resp.status_code == 200
            body = resp.text
            assert "error" in body.lower() or "alert" in body.lower()
        finally:
            chat_module._create_runtime = original_create
            test_app.dependency_overrides.clear()


class TestChatPageSidebar:
    """Tests for chat page sidebar and conversation UI elements."""

    async def test_chat_page_has_conversation_sidebar(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat")
        assert resp.status_code == 200
        assert 'id="conversation-list"' in resp.text
        assert "Conversations" in resp.text

    async def test_chat_page_has_hidden_conversation_id(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat")
        assert resp.status_code == 200
        assert 'name="conversation_id"' in resp.text

    async def test_chat_page_has_new_conversation_button(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat")
        assert resp.status_code == 200
        assert "New" in resp.text


class TestConversationEndpoints:
    """Tests for conversation CRUD endpoints."""

    async def test_list_conversations_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/chat/conversations")
        assert resp.status_code == 303

    async def test_list_conversations_returns_html(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import src.web.routes.chat as chat_module

        mock_service = AsyncMock()
        mock_service.list_conversations.return_value = []
        original = chat_module._create_conversation_service
        chat_module._create_conversation_service = lambda *a, **kw: mock_service

        try:
            with AuthCookies(client, admin_token):
                resp = await client.get("/web/chat/conversations")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
        finally:
            chat_module._create_conversation_service = original

    async def test_get_conversation_returns_messages(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import uuid

        import src.web.routes.chat as chat_module
        from src.db.models import Conversation, Message

        cid = uuid.UUID("00000000-0000-0000-0000-000000000010")
        uid = uuid.UUID("00000000-0000-0000-0000-000000000001")
        mock_service = AsyncMock()
        mock_service.get_conversation.return_value = Conversation(id=cid, user_id=uid, title="Test")
        mock_service.get_messages.return_value = [
            Message(conversation_id=cid, role="user", content="Hi"),
            Message(conversation_id=cid, role="assistant", content="Hello"),
        ]
        original = chat_module._create_conversation_service
        chat_module._create_conversation_service = lambda *a, **kw: mock_service

        try:
            with AuthCookies(client, admin_token):
                resp = await client.get(f"/web/chat/conversations/{cid}")
            assert resp.status_code == 200
            assert "Hi" in resp.text
            assert "Hello" in resp.text
        finally:
            chat_module._create_conversation_service = original

    async def test_create_conversation_returns_redirect(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import uuid

        import src.web.routes.chat as chat_module
        from src.db.models import Conversation

        cid = uuid.UUID("00000000-0000-0000-0000-000000000099")
        uid = uuid.UUID("00000000-0000-0000-0000-000000000001")
        mock_service = AsyncMock()
        mock_service.create_conversation.return_value = Conversation(id=cid, user_id=uid)
        original = chat_module._create_conversation_service
        chat_module._create_conversation_service = lambda *a, **kw: mock_service

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post("/web/chat/conversations/new")
            # Should redirect or return HX-Redirect header
            assert resp.status_code in (200, 303)
        finally:
            chat_module._create_conversation_service = original

    async def test_delete_conversation(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import uuid

        import src.web.routes.chat as chat_module

        cid = uuid.UUID("00000000-0000-0000-0000-000000000010")
        mock_service = AsyncMock()
        mock_service.delete_conversation.return_value = True
        original = chat_module._create_conversation_service
        chat_module._create_conversation_service = lambda *a, **kw: mock_service

        try:
            with AuthCookies(client, admin_token):
                resp = await client.delete(f"/web/chat/conversations/{cid}")
            assert resp.status_code == 200
        finally:
            chat_module._create_conversation_service = original


class TestChatSendWithConversation:
    """Tests for POST /web/chat/send with conversation persistence."""

    async def test_send_with_conversation_id_calls_runtime_with_history(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import uuid

        import src.web.routes.chat as chat_module
        from src.db.models import Conversation, Message

        cid = uuid.UUID("00000000-0000-0000-0000-000000000010")
        uid = uuid.UUID("00000000-0000-0000-0000-000000000001")

        mock_runtime = AsyncMock()
        mock_runtime.run.return_value = AgentResult(
            answer="I remember.",
            steps=[AgentStep(thought="Using history.")],
            total_steps=1,
            stopped_by_max_steps=False,
        )

        mock_service = AsyncMock()
        mock_service.get_conversation.return_value = Conversation(
            id=cid, user_id=uid, title="Existing"
        )
        mock_service.get_messages.return_value = [
            Message(conversation_id=cid, role="user", content="Previous question"),
            Message(conversation_id=cid, role="assistant", content="Previous answer"),
        ]
        mock_service.add_message.return_value = Message(
            conversation_id=cid, role="user", content="Follow up"
        )

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        original_create = chat_module._create_runtime
        original_service = chat_module._create_conversation_service

        chat_module._create_runtime = lambda *a, **kw: mock_runtime
        chat_module._create_conversation_service = lambda *a, **kw: mock_service

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/chat/send",
                    data={"message": "Follow up", "conversation_id": str(cid)},
                )
            assert resp.status_code == 200
            assert "I remember." in resp.text
            # Verify runtime.run was called with conversation_history
            mock_runtime.run.assert_awaited_once()
            call_kwargs = mock_runtime.run.call_args
            assert call_kwargs[1].get("conversation_history") is not None
            assert len(call_kwargs[1]["conversation_history"]) == 2
        finally:
            chat_module._create_runtime = original_create
            chat_module._create_conversation_service = original_service
            test_app.dependency_overrides.clear()


class TestChatStream:
    """Tests for GET /web/chat/stream SSE endpoint."""

    async def test_stream_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/chat/stream?message=hello")
        assert resp.status_code == 303

    async def test_stream_returns_event_stream(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import uuid

        import src.web.routes.chat as chat_module
        from src.agent.runtime import AgentAnswerEvent, AgentStepEvent
        from src.agent.state import AgentStep as _AgentStep
        from src.db.models import Conversation

        async def _mock_streaming(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            yield AgentStepEvent(
                step=_AgentStep(thought="thinking"),
                step_number=1,
            )
            yield AgentAnswerEvent(
                answer="The answer is 42.",
                total_steps=1,
                stopped_by_max_steps=False,
                sources=[],
            )

        mock_runtime = AsyncMock()
        mock_runtime.run_streaming = _mock_streaming

        mock_service = AsyncMock()
        cid = uuid.UUID("00000000-0000-0000-0000-000000000050")
        uid = uuid.UUID("00000000-0000-0000-0000-000000000001")
        mock_service.create_conversation.return_value = Conversation(id=cid, user_id=uid)
        mock_service.get_conversation.return_value = None
        mock_service.get_messages.return_value = []
        mock_service.add_message.return_value = None
        mock_service.update_title.return_value = None

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        original_create = chat_module._create_runtime
        original_service = chat_module._create_conversation_service

        chat_module._create_runtime = lambda *a, **kw: mock_runtime
        chat_module._create_conversation_service = lambda *a, **kw: mock_service

        try:
            with AuthCookies(client, admin_token):
                resp = await client.get("/web/chat/stream?message=What+is+42")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = resp.text
            assert "event: agent-step" in body
            assert "event: agent-answer" in body
            assert "event: done" in body
        finally:
            chat_module._create_runtime = original_create
            chat_module._create_conversation_service = original_service
            test_app.dependency_overrides.clear()

    async def test_stream_empty_message_returns_error_event(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/chat/stream?message=")
        assert resp.status_code == 200
        assert "event: error" in resp.text

"""Tests for web chat routes (Agent-integrated)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
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


class TestChatSend:
    """Tests for POST /web/chat/send (Agent-integrated)."""

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

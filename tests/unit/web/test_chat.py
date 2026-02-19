"""Tests for web chat routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.config import Settings
from src.llm.providers.base import LLMChunk, LLMResponse, TokenUsage
from src.main import create_app

from tests.unit.web.conftest import auth_cookies


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
        resp = await client.get("/web/chat", cookies=auth_cookies(admin_token))
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Chat" in resp.text


class TestChatSend:
    """Tests for POST /web/chat/send."""

    async def test_send_returns_message_fragment(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = LLMResponse(
            content="Hello from the LLM!",
            model="test-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        from src.api.dependencies import get_llm_model, get_llm_provider

        test_app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        resp = await client.post(
            "/web/chat/send",
            data={"message": "Hello"},
            cookies=auth_cookies(admin_token),
        )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Hello from the LLM!" in resp.text

        test_app.dependency_overrides.clear()

    async def test_send_empty_message_returns_error(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        resp = await client.post(
            "/web/chat/send",
            data={"message": ""},
            cookies=auth_cookies(admin_token),
        )
        assert resp.status_code == 200
        body = resp.text
        assert "alert" in body.lower() or "error" in body.lower()


class TestChatStream:
    """Tests for POST /web/chat/stream (SSE)."""

    async def test_stream_returns_sse_response(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        async def mock_stream(**kwargs: object) -> AsyncGenerator[LLMChunk, None]:
            yield LLMChunk(content="Hello", finish_reason=None)
            yield LLMChunk(content=" world", finish_reason="stop")

        mock_provider = AsyncMock()
        mock_provider.stream = mock_stream

        from src.api.dependencies import get_llm_model, get_llm_provider

        test_app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        resp = await client.post(
            "/web/chat/stream",
            data={"message": "Hello"},
            cookies=auth_cookies(admin_token),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "Hello" in body
        assert "world" in body

        test_app.dependency_overrides.clear()

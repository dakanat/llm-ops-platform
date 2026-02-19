"""Tests for web Agent routes."""

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


class TestAgentPage:
    """Tests for GET /web/agent."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/agent")
        assert resp.status_code == 303

    async def test_authenticated_returns_agent_page(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        resp = await client.get("/web/agent", cookies=auth_cookies(admin_token))
        assert resp.status_code == 200
        assert "Agent" in resp.text


class TestAgentRun:
    """Tests for POST /web/agent/run."""

    async def test_run_returns_result_fragment(
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
            ],
            total_steps=1,
            stopped_by_max_steps=False,
        )

        from src.agent.tools.registry import ToolRegistry
        from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry

        test_app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"
        test_app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

        import src.web.routes.agent as agent_module

        original_create = agent_module._create_runtime

        def _mock_create(*args: object, **kwargs: object) -> AsyncMock:
            return mock_runtime

        agent_module._create_runtime = _mock_create

        try:
            resp = await client.post(
                "/web/agent/run",
                data={"query": "What is 6 * 7?"},
                cookies=auth_cookies(admin_token),
            )
            assert resp.status_code == 200
            assert "42" in resp.text
            assert "calculator" in resp.text
        finally:
            agent_module._create_runtime = original_create
            test_app.dependency_overrides.clear()

    async def test_empty_query_returns_error(self, client: AsyncClient, admin_token: str) -> None:
        resp = await client.post(
            "/web/agent/run",
            data={"query": ""},
            cookies=auth_cookies(admin_token),
        )
        assert resp.status_code == 200
        body = resp.text
        assert "alert" in body.lower() or "error" in body.lower() or "empty" in body.lower()

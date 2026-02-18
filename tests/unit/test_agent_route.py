"""Tests for POST /agent/run endpoint."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from src.agent import AgentError
from src.agent.runtime import AgentResult
from src.agent.state import AgentStep
from src.agent.tools.registry import ToolRegistry
from src.api.middleware.auth import TokenPayload
from src.main import app


def _make_agent_result(
    answer: str = "The answer is 42.",
    steps: list[AgentStep] | None = None,
    total_steps: int = 1,
    stopped_by_max_steps: bool = False,
) -> AgentResult:
    """Create a test AgentResult."""
    return AgentResult(
        answer=answer,
        steps=steps
        or [
            AgentStep(
                thought="I need to calculate this.",
                action="calculator",
                action_input="6 * 7",
                observation="42",
                is_error=False,
            ),
        ],
        total_steps=total_steps,
        stopped_by_max_steps=stopped_by_max_steps,
    )


def _admin_user() -> TokenPayload:
    """Return a TokenPayload with admin role."""
    return TokenPayload(
        sub="user-1",
        email="admin@example.com",
        role="admin",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


def _viewer_user() -> TokenPayload:
    """Return a TokenPayload with viewer role."""
    return TokenPayload(
        sub="user-2",
        email="viewer@example.com",
        role="viewer",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


def _override_dependencies(
    user: TokenPayload | None = None,
) -> None:
    """Set FastAPI dependency overrides for agent route tests."""
    from src.api.dependencies import get_llm_model, get_llm_provider, get_tool_registry
    from src.api.middleware.auth import get_current_user

    app.dependency_overrides[get_llm_provider] = lambda: AsyncMock()
    app.dependency_overrides[get_llm_model] = lambda: "test-model"
    app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()
    app.dependency_overrides[get_current_user] = lambda: user or _admin_user()


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    """Clear dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class TestAgentRunRoute:
    """POST /agent/run のテスト。"""

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_200_with_valid_query(self, mock_runtime_cls: AsyncMock) -> None:
        """有効なクエリで 200 が返ること。"""
        mock_runtime_cls.return_value.run = AsyncMock(return_value=_make_agent_result())
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.status_code == 200

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_answer_from_runtime(self, mock_runtime_cls: AsyncMock) -> None:
        """Runtime の回答がレスポンスに含まれること。"""
        mock_runtime_cls.return_value.run = AsyncMock(
            return_value=_make_agent_result(answer="The result is 42.")
        )
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.json()["answer"] == "The result is 42."

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_steps(self, mock_runtime_cls: AsyncMock) -> None:
        """ステップリストがレスポンスに含まれること。"""
        steps = [
            AgentStep(
                thought="Let me calculate.",
                action="calculator",
                action_input="6 * 7",
                observation="42",
                is_error=False,
            ),
        ]
        mock_runtime_cls.return_value.run = AsyncMock(return_value=_make_agent_result(steps=steps))
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        resp_steps = response.json()["steps"]
        assert len(resp_steps) == 1
        assert resp_steps[0]["thought"] == "Let me calculate."
        assert resp_steps[0]["action"] == "calculator"
        assert resp_steps[0]["action_input"] == "6 * 7"
        assert resp_steps[0]["observation"] == "42"
        assert resp_steps[0]["is_error"] is False

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_total_steps(self, mock_runtime_cls: AsyncMock) -> None:
        """total_steps がレスポンスに含まれること。"""
        mock_runtime_cls.return_value.run = AsyncMock(
            return_value=_make_agent_result(total_steps=3)
        )
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.json()["total_steps"] == 3

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_stopped_by_max_steps_false(self, mock_runtime_cls: AsyncMock) -> None:
        """正常完了時に stopped_by_max_steps が false であること。"""
        mock_runtime_cls.return_value.run = AsyncMock(
            return_value=_make_agent_result(stopped_by_max_steps=False)
        )
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.json()["stopped_by_max_steps"] is False

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_stopped_by_max_steps_true(self, mock_runtime_cls: AsyncMock) -> None:
        """最大ステップ数で停止時に stopped_by_max_steps が true であること。"""
        mock_runtime_cls.return_value.run = AsyncMock(
            return_value=_make_agent_result(stopped_by_max_steps=True)
        )
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.json()["stopped_by_max_steps"] is True

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_passes_query_to_runtime(self, mock_runtime_cls: AsyncMock) -> None:
        """クエリが Runtime.run() に渡されること。"""
        mock_runtime_cls.return_value.run = AsyncMock(return_value=_make_agent_result())
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        mock_runtime_cls.return_value.run.assert_called_once_with("What is 6 * 7?")

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_passes_max_steps_to_runtime_constructor(
        self, mock_runtime_cls: AsyncMock
    ) -> None:
        """max_steps が AgentRuntime コンストラクタに渡されること。"""
        mock_runtime_cls.return_value.run = AsyncMock(return_value=_make_agent_result())
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/agent/run", json={"query": "What is 6 * 7?", "max_steps": 5})

        _, kwargs = mock_runtime_cls.call_args
        assert kwargs["max_steps"] == 5

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_defaults_max_steps_to_10(self, mock_runtime_cls: AsyncMock) -> None:
        """max_steps 省略時にデフォルト 10 が渡されること。"""
        mock_runtime_cls.return_value.run = AsyncMock(return_value=_make_agent_result())
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        _, kwargs = mock_runtime_cls.call_args
        assert kwargs["max_steps"] == 10

    async def test_returns_422_for_missing_query(self) -> None:
        """query フィールドがない場合に 422 が返ること。"""
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={})

        assert response.status_code == 422

    async def test_returns_422_for_empty_query(self) -> None:
        """空文字の query で 422 が返ること。"""
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": ""})

        assert response.status_code == 422

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_502_when_agent_fails(self, mock_runtime_cls: AsyncMock) -> None:
        """AgentError 発生時に 502 が返ること。"""
        mock_runtime_cls.return_value.run = AsyncMock(side_effect=AgentError("LLM call failed"))
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.status_code == 502

    async def test_returns_401_without_auth(self) -> None:
        """認証なしで 401 が返ること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.status_code in (401, 403)

    @patch("src.api.routes.agent.AgentRuntime")
    async def test_returns_403_for_viewer_role(self, mock_runtime_cls: AsyncMock) -> None:
        """viewer ロールで 403 が返ること。"""
        mock_runtime_cls.return_value.run = AsyncMock(return_value=_make_agent_result())
        _override_dependencies(user=_viewer_user())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/agent/run", json={"query": "What is 6 * 7?"})

        assert response.status_code == 403

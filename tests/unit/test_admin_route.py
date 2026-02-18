"""Tests for GET /admin/metrics and GET /admin/costs endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.middleware.auth import TokenPayload
from src.main import app
from src.monitoring.cost_tracker import CostReport, CostTracker, ModelCostSummary


def _admin_user() -> TokenPayload:
    """Return a TokenPayload with admin role."""
    return TokenPayload(
        sub="user-1",
        email="admin@example.com",
        role="admin",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


def _user_role() -> TokenPayload:
    """Return a TokenPayload with user role."""
    return TokenPayload(
        sub="user-2",
        email="user@example.com",
        role="user",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


def _viewer_user() -> TokenPayload:
    """Return a TokenPayload with viewer role."""
    return TokenPayload(
        sub="user-3",
        email="viewer@example.com",
        role="viewer",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


def _mock_cost_tracker(
    total_cost: float = 5.0,
    model_costs: dict[str, ModelCostSummary] | None = None,
    alert_triggered: bool = False,
) -> CostTracker:
    """Create a mock CostTracker."""
    tracker = AsyncMock(spec=CostTracker)
    report = CostReport(
        total_cost=total_cost,
        model_costs=model_costs or {"gpt-4": ModelCostSummary(cost=5.0, requests=10)},
    )
    tracker.get_cost_report.return_value = report
    tracker.is_alert_triggered.return_value = alert_triggered
    return tracker


def _override_dependencies(
    user: TokenPayload | None = None,
    cost_tracker: CostTracker | None = None,
) -> None:
    """Set FastAPI dependency overrides for admin route tests."""
    from src.api.dependencies import get_cost_tracker
    from src.api.middleware.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: user or _admin_user()
    app.dependency_overrides[get_cost_tracker] = lambda: cost_tracker or _mock_cost_tracker()


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    """Clear dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class TestAdminMetricsRoute:
    """GET /admin/metrics のテスト。"""

    async def test_metrics_returns_200(self) -> None:
        """200 が返ること。"""
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/metrics")

        assert response.status_code == 200

    async def test_metrics_returns_prometheus_content_type(self) -> None:
        """Prometheus 形式の Content-Type が返ること。"""
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/metrics")

        assert "text/plain" in response.headers["content-type"]

    async def test_metrics_returns_401_without_auth(self) -> None:
        """認証なしで 401 が返ること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/metrics")

        assert response.status_code in (401, 403)

    async def test_metrics_returns_403_for_user_role(self) -> None:
        """user ロールで 403 が返ること。"""
        _override_dependencies(user=_user_role())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/metrics")

        assert response.status_code == 403


class TestAdminCostsRoute:
    """GET /admin/costs のテスト。"""

    async def test_costs_returns_200(self) -> None:
        """200 が返ること。"""
        _override_dependencies()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/costs")

        assert response.status_code == 200

    async def test_costs_returns_total_cost(self) -> None:
        """total_cost がレスポンスに含まれること。"""
        tracker = _mock_cost_tracker(total_cost=12.5)
        _override_dependencies(cost_tracker=tracker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/costs")

        assert response.json()["total_cost"] == 12.5

    async def test_costs_returns_model_costs(self) -> None:
        """model_costs がレスポンスに含まれること。"""
        model_costs = {
            "gpt-4": ModelCostSummary(cost=8.0, requests=20),
            "gpt-3.5": ModelCostSummary(cost=2.0, requests=50),
        }
        tracker = _mock_cost_tracker(total_cost=10.0, model_costs=model_costs)
        _override_dependencies(cost_tracker=tracker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/costs")

        resp_costs = response.json()["model_costs"]
        assert resp_costs["gpt-4"]["cost"] == 8.0
        assert resp_costs["gpt-4"]["requests"] == 20
        assert resp_costs["gpt-3.5"]["cost"] == 2.0
        assert resp_costs["gpt-3.5"]["requests"] == 50

    async def test_costs_returns_alert_triggered_false(self) -> None:
        """アラート未発生時に alert_triggered が false であること。"""
        tracker = _mock_cost_tracker(alert_triggered=False)
        _override_dependencies(cost_tracker=tracker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/costs")

        assert response.json()["alert_triggered"] is False

    async def test_costs_returns_alert_triggered_true(self) -> None:
        """アラート発生時に alert_triggered が true であること。"""
        tracker = _mock_cost_tracker(alert_triggered=True)
        _override_dependencies(cost_tracker=tracker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/costs")

        assert response.json()["alert_triggered"] is True

    async def test_costs_returns_401_without_auth(self) -> None:
        """認証なしで 401 が返ること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/costs")

        assert response.status_code in (401, 403)

    async def test_costs_returns_403_for_viewer_role(self) -> None:
        """viewer ロールで 403 が返ること。"""
        _override_dependencies(user=_viewer_user())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/costs")

        assert response.status_code == 403

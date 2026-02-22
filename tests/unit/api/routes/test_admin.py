"""Tests for GET /admin/metrics, GET /admin/costs, and GET /admin/audit-logs endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import AsyncClient
from src.api.middleware.auth import TokenPayload
from src.monitoring.cost_tracker import CostReport, CostTracker, ModelCostSummary


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
    app: FastAPI,
    user: TokenPayload | None = None,
    cost_tracker: CostTracker | None = None,
    admin_user: TokenPayload | None = None,
) -> None:
    """Set FastAPI dependency overrides for admin route tests."""
    from src.api.dependencies import get_cost_tracker
    from src.api.middleware.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: user or admin_user
    app.dependency_overrides[get_cost_tracker] = lambda: cost_tracker or _mock_cost_tracker()


class TestAdminMetricsRoute:
    """GET /admin/metrics のテスト。"""

    async def test_metrics_returns_200(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """200 が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.get("/admin/metrics")

        assert response.status_code == 200

    async def test_metrics_returns_prometheus_content_type(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """Prometheus 形式の Content-Type が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.get("/admin/metrics")

        assert "text/plain" in response.headers["content-type"]

    async def test_metrics_returns_401_without_auth(self, client: AsyncClient) -> None:
        """認証なしで 401 が返ること。"""
        response = await client.get("/admin/metrics")

        assert response.status_code in (401, 403)

    async def test_metrics_returns_403_for_user_role(
        self, client: AsyncClient, user_role: TokenPayload, test_app: FastAPI
    ) -> None:
        """user ロールで 403 が返ること。"""
        _override_dependencies(test_app, user=user_role)
        response = await client.get("/admin/metrics")

        assert response.status_code == 403


class TestAdminCostsRoute:
    """GET /admin/costs のテスト。"""

    async def test_costs_returns_200(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """200 が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.get("/admin/costs")

        assert response.status_code == 200

    async def test_costs_returns_total_cost(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """total_cost がレスポンスに含まれること。"""
        tracker = _mock_cost_tracker(total_cost=12.5)
        _override_dependencies(test_app, cost_tracker=tracker, admin_user=admin_user)
        response = await client.get("/admin/costs")

        assert response.json()["total_cost"] == 12.5

    async def test_costs_returns_model_costs(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """model_costs がレスポンスに含まれること。"""
        model_costs = {
            "gpt-4": ModelCostSummary(cost=8.0, requests=20),
            "gpt-3.5": ModelCostSummary(cost=2.0, requests=50),
        }
        tracker = _mock_cost_tracker(total_cost=10.0, model_costs=model_costs)
        _override_dependencies(test_app, cost_tracker=tracker, admin_user=admin_user)
        response = await client.get("/admin/costs")

        resp_costs = response.json()["model_costs"]
        assert resp_costs["gpt-4"]["cost"] == 8.0
        assert resp_costs["gpt-4"]["requests"] == 20
        assert resp_costs["gpt-3.5"]["cost"] == 2.0
        assert resp_costs["gpt-3.5"]["requests"] == 50

    async def test_costs_returns_alert_triggered_false(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """アラート未発生時に alert_triggered が false であること。"""
        tracker = _mock_cost_tracker(alert_triggered=False)
        _override_dependencies(test_app, cost_tracker=tracker, admin_user=admin_user)
        response = await client.get("/admin/costs")

        assert response.json()["alert_triggered"] is False

    async def test_costs_returns_alert_triggered_true(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """アラート発生時に alert_triggered が true であること。"""
        tracker = _mock_cost_tracker(alert_triggered=True)
        _override_dependencies(test_app, cost_tracker=tracker, admin_user=admin_user)
        response = await client.get("/admin/costs")

        assert response.json()["alert_triggered"] is True

    async def test_costs_returns_401_without_auth(self, client: AsyncClient) -> None:
        """認証なしで 401 が返ること。"""
        response = await client.get("/admin/costs")

        assert response.status_code in (401, 403)

    async def test_costs_returns_403_for_viewer_role(
        self, client: AsyncClient, viewer_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """viewer ロールで 403 が返ること。"""
        _override_dependencies(test_app, user=viewer_user)
        response = await client.get("/admin/costs")

        assert response.status_code == 403


def _make_audit_session(
    *,
    audit_logs: list[object] | None = None,
    total: int = 0,
) -> AsyncMock:
    """Create a mock session for audit log queries."""

    from unittest.mock import MagicMock

    session = AsyncMock()
    session.add = MagicMock()

    # exec is called twice: once for count, once for items
    count_result = MagicMock()
    count_result.one.return_value = total

    items_result = MagicMock()
    items_result.all.return_value = audit_logs or []

    session.exec.side_effect = [count_result, items_result]
    return session


def _override_audit_deps(
    app: FastAPI,
    *,
    user: TokenPayload | None = None,
    session: AsyncMock | None = None,
) -> None:
    """Set dependency overrides for audit log tests."""
    from src.api.dependencies import get_cost_tracker
    from src.api.middleware.auth import get_current_user
    from src.db.session import get_session

    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    if session is not None:
        app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_cost_tracker] = lambda: _mock_cost_tracker()


class TestAdminAuditLogs:
    """GET /admin/audit-logs のテスト。"""

    async def test_audit_logs_returns_200_for_admin(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """admin ユーザーで 200 が返ること。"""
        session = _make_audit_session(total=0)
        _override_audit_deps(test_app, user=admin_user, session=session)

        response = await client.get("/admin/audit-logs")

        assert response.status_code == 200

    async def test_audit_logs_returns_paginated_response(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """items, total, page, page_size フィールドが含まれること。"""
        session = _make_audit_session(total=0)
        _override_audit_deps(test_app, user=admin_user, session=session)

        response = await client.get("/admin/audit-logs")

        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    async def test_audit_logs_returns_403_for_viewer(
        self, client: AsyncClient, viewer_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """viewer で 403 が返ること。"""
        session = _make_audit_session(total=0)
        _override_audit_deps(test_app, user=viewer_user, session=session)

        response = await client.get("/admin/audit-logs")

        assert response.status_code == 403

    async def test_audit_logs_returns_401_without_auth(self, client: AsyncClient) -> None:
        """未認証で 401 が返ること。"""
        response = await client.get("/admin/audit-logs")

        assert response.status_code in (401, 403)

    async def test_audit_logs_returns_items_with_expected_fields(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """レスポンスの items に必要なフィールドが含まれること。"""
        import uuid
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        mock_log = MagicMock()
        mock_log.id = uuid.uuid4()
        mock_log.user_id = uuid.uuid4()
        mock_log.action = "create"
        mock_log.resource_type = "eval_dataset"
        mock_log.resource_id = str(uuid.uuid4())
        mock_log.details = {"name": "ds-1"}
        mock_log.created_at = datetime.now(UTC)

        session = _make_audit_session(audit_logs=[mock_log], total=1)
        _override_audit_deps(test_app, user=admin_user, session=session)

        response = await client.get("/admin/audit-logs")

        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["action"] == "create"
        assert item["resource_type"] == "eval_dataset"
        assert item["details"]["name"] == "ds-1"

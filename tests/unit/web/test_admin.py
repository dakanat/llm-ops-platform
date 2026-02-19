"""Tests for web Admin routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.api.middleware.auth import create_access_token
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
    """Return a valid JWT token with admin role."""
    return create_access_token(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        email="admin@example.com",
        role="admin",
    )


@pytest.fixture
def viewer_token() -> str:
    """Return a valid JWT token with viewer role."""
    return create_access_token(
        user_id=UUID("00000000-0000-0000-0000-000000000003"),
        email="viewer@example.com",
        role="viewer",
    )


class TestAdminPage:
    """Tests for GET /web/admin."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/admin")
        assert resp.status_code == 303

    async def test_admin_can_access(self, client: AsyncClient, admin_token: str) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/admin")
        assert resp.status_code == 200
        assert "Admin" in resp.text

    async def test_viewer_gets_forbidden(self, client: AsyncClient, viewer_token: str) -> None:
        with AuthCookies(client, viewer_token):
            resp = await client.get("/web/admin")
        assert (
            resp.status_code == 403 or "Forbidden" in resp.text or "permission" in resp.text.lower()
        )


class TestAdminCosts:
    """Tests for GET /web/admin/costs."""

    async def test_admin_gets_cost_report(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        mock_tracker = MagicMock()
        mock_tracker.get_cost_report.return_value = {
            "total_cost": 1.5,
            "model_costs": {
                "gemini-2.5-flash-lite": {"cost": 1.5, "requests": 100},
            },
        }
        mock_tracker.is_alert_triggered.return_value = False

        from src.api.dependencies import get_cost_tracker

        test_app.dependency_overrides[get_cost_tracker] = lambda: mock_tracker

        with AuthCookies(client, admin_token):
            resp = await client.get("/web/admin/costs")
        assert resp.status_code == 200
        assert "1.5" in resp.text or "$1.50" in resp.text
        assert "gemini" in resp.text.lower()

        test_app.dependency_overrides.clear()

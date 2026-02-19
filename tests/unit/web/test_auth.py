"""Tests for web authentication routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.api.middleware.auth import (
    create_access_token,
    hash_password,
)
from src.config import Settings
from src.main import create_app


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


def _make_mock_session(user: MagicMock | None = None) -> AsyncMock:
    """Create a mock session that returns the given user."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = user
    mock_session.exec.return_value = mock_result
    return mock_session


class TestLoginPage:
    """Tests for GET /web/login."""

    async def test_login_page_returns_html(self, client: AsyncClient) -> None:
        resp = await client.get("/web/login")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_login_page_contains_form(self, client: AsyncClient) -> None:
        resp = await client.get("/web/login")
        body = resp.text
        assert "<form" in body
        assert 'name="email"' in body
        assert 'name="password"' in body


class TestLoginPost:
    """Tests for POST /web/login."""

    async def test_successful_login_sets_cookie_and_redirects(self, client: AsyncClient) -> None:
        mock_user = MagicMock()
        mock_user.id = UUID("00000000-0000-0000-0000-000000000001")
        mock_user.email = "admin@example.com"
        mock_user.role = "admin"
        mock_user.hashed_password = hash_password("correct-password")
        mock_user.is_active = True

        mock_session = _make_mock_session(mock_user)

        async def _fake_get_session() -> AsyncMock:
            return mock_session

        with patch("src.web.routes.auth._get_session", side_effect=_fake_get_session):
            resp = await client.post(
                "/web/login",
                data={"email": "admin@example.com", "password": "correct-password"},
            )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/web/chat"
        assert "access_token" in resp.cookies

    async def test_invalid_credentials_returns_login_with_error(self, client: AsyncClient) -> None:
        mock_session = _make_mock_session(None)

        async def _fake_get_session() -> AsyncMock:
            return mock_session

        with patch("src.web.routes.auth._get_session", side_effect=_fake_get_session):
            resp = await client.post(
                "/web/login",
                data={"email": "wrong@example.com", "password": "wrong"},
            )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "Invalid" in body

    async def test_inactive_user_cannot_login(self, client: AsyncClient) -> None:
        mock_user = MagicMock()
        mock_user.id = UUID("00000000-0000-0000-0000-000000000001")
        mock_user.email = "inactive@example.com"
        mock_user.role = "user"
        mock_user.hashed_password = hash_password("password")
        mock_user.is_active = False

        mock_session = _make_mock_session(mock_user)

        async def _fake_get_session() -> AsyncMock:
            return mock_session

        with patch("src.web.routes.auth._get_session", side_effect=_fake_get_session):
            resp = await client.post(
                "/web/login",
                data={"email": "inactive@example.com", "password": "password"},
            )
        assert resp.status_code == 200
        body = resp.text
        assert "Invalid" in body


class TestLogout:
    """Tests for POST /web/logout."""

    async def test_logout_clears_cookie_and_redirects(self, client: AsyncClient) -> None:
        token = create_access_token(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            email="admin@example.com",
            role="admin",
        )
        resp = await client.post(
            "/web/logout",
            cookies={"access_token": token},
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/web/login"
        cookie_header = resp.headers.get("set-cookie", "")
        assert "access_token" in cookie_header


class TestRootRedirect:
    """Tests for GET / redirect."""

    async def test_root_redirects_to_web_chat(self, client: AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code in (301, 302, 303, 307, 308)
        assert "/web/chat" in resp.headers["location"]

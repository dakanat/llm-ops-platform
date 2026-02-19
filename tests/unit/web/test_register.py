"""Tests for web registration routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
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


def _make_mock_session() -> AsyncMock:
    """Create a mock session for registration tests."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_session.exec.return_value = mock_result
    return mock_session


class TestRegisterPage:
    """Tests for GET /web/register."""

    async def test_register_page_returns_html(self, client: AsyncClient) -> None:
        """登録ページが 200 + HTML を返すこと。"""
        resp = await client.get("/web/register")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_register_page_contains_form_fields(self, client: AsyncClient) -> None:
        """登録フォームに email, name, password フィールドがあること。"""
        resp = await client.get("/web/register")
        body = resp.text
        assert "<form" in body
        assert 'name="email"' in body
        assert 'name="name"' in body
        assert 'name="password"' in body


class TestRegisterPost:
    """Tests for POST /web/register."""

    async def test_successful_registration_redirects_to_chat(self, client: AsyncClient) -> None:
        """正常登録で Cookie 設定 → 303 /web/chat リダイレクトすること。"""
        mock_session = _make_mock_session()

        async def _fake_get_session() -> AsyncMock:
            return mock_session

        with patch("src.web.routes.auth._get_session", side_effect=_fake_get_session):
            resp = await client.post(
                "/web/register",
                data={
                    "email": "newuser@example.com",
                    "name": "New User",
                    "password": "securepassword123",
                },
            )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/web/chat"
        assert "access_token" in resp.cookies

    async def test_duplicate_email_shows_error(self, client: AsyncClient) -> None:
        """メール重複で エラーメッセージ付きフォームが返ること。"""
        from sqlalchemy.exc import IntegrityError

        mock_session = _make_mock_session()
        mock_session.commit.side_effect = IntegrityError("duplicate", params={}, orig=Exception())

        async def _fake_get_session() -> AsyncMock:
            return mock_session

        with patch("src.web.routes.auth._get_session", side_effect=_fake_get_session):
            resp = await client.post(
                "/web/register",
                data={
                    "email": "existing@example.com",
                    "name": "Existing User",
                    "password": "securepassword123",
                },
            )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "already" in resp.text.lower()

    async def test_validation_error_shows_error_message(self, client: AsyncClient) -> None:
        """バリデーション失敗でエラーメッセージ付きフォームが返ること。"""
        mock_session = _make_mock_session()

        async def _fake_get_session() -> AsyncMock:
            return mock_session

        with patch("src.web.routes.auth._get_session", side_effect=_fake_get_session):
            resp = await client.post(
                "/web/register",
                data={
                    "email": "newuser@example.com",
                    "name": "",
                    "password": "securepassword123",
                },
            )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_short_password_shows_error(self, client: AsyncClient) -> None:
        """パスワード 8 文字未満でエラーメッセージ付きフォームが返ること。"""
        mock_session = _make_mock_session()

        async def _fake_get_session() -> AsyncMock:
            return mock_session

        with patch("src.web.routes.auth._get_session", side_effect=_fake_get_session):
            resp = await client.post(
                "/web/register",
                data={
                    "email": "newuser@example.com",
                    "name": "New User",
                    "password": "short",
                },
            )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

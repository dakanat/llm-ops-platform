"""Tests for auth registration API endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import bcrypt
from fastapi import FastAPI
from httpx import AsyncClient
from src.db.session import get_session


def _make_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    result_proxy = MagicMock()
    result_proxy.all.return_value = []
    result_proxy.first.return_value = None
    session.exec.return_value = result_proxy
    session.get.return_value = None
    return session


def _override_deps(app: FastAPI, *, session: AsyncMock) -> None:
    """Set dependency overrides for auth tests (no auth required)."""
    app.dependency_overrides[get_session] = lambda: session


_VALID_PAYLOAD = {
    "email": "newuser@example.com",
    "name": "New User",
    "password": "securepassword123",
}


class TestRegisterUser:
    """POST /auth/register のテスト。"""

    async def test_returns_201_on_success(self, client: AsyncClient, test_app: FastAPI) -> None:
        """正常登録で 201 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        response = await client.post("/auth/register", json=_VALID_PAYLOAD)

        assert response.status_code == 201

    async def test_response_contains_required_fields(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """レスポンスに id, email, name, role, access_token が含まれること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        response = await client.post("/auth/register", json=_VALID_PAYLOAD)

        data = response.json()
        assert "id" in data
        assert data["email"] == "newuser@example.com"
        assert data["name"] == "New User"
        assert data["role"] == "user"
        assert "access_token" in data

    async def test_role_is_always_user(self, client: AsyncClient, test_app: FastAPI) -> None:
        """リクエストに role: admin を送っても role は常に user になること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        payload = {**_VALID_PAYLOAD, "role": "admin"}
        response = await client.post("/auth/register", json=payload)

        assert response.json()["role"] == "user"

    async def test_password_is_hashed(self, client: AsyncClient, test_app: FastAPI) -> None:
        """パスワードがハッシュ化されて保存されること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        await client.post("/auth/register", json=_VALID_PAYLOAD)

        added_user = session.add.call_args[0][0]
        assert added_user.hashed_password != _VALID_PAYLOAD["password"]
        assert bcrypt.checkpw(
            _VALID_PAYLOAD["password"].encode(),
            added_user.hashed_password.encode(),
        )

    async def test_calls_session_add_and_commit(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """session.add と session.commit が呼ばれること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        await client.post("/auth/register", json=_VALID_PAYLOAD)

        assert session.add.called
        assert session.commit.called

    async def test_returns_409_on_duplicate_email(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """メール重複時に 409 Conflict が返ること。"""
        from sqlalchemy.exc import IntegrityError

        session = _make_session()
        session.commit.side_effect = IntegrityError("duplicate", params={}, orig=Exception())
        _override_deps(test_app, session=session)

        response = await client.post("/auth/register", json=_VALID_PAYLOAD)

        assert response.status_code == 409

    async def test_returns_422_for_invalid_email(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """不正なメール形式で 422 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        payload = {**_VALID_PAYLOAD, "email": "not-an-email"}
        response = await client.post("/auth/register", json=payload)

        assert response.status_code == 422

    async def test_returns_422_for_short_password(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """パスワード 8 文字未満で 422 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        payload = {**_VALID_PAYLOAD, "password": "short"}
        response = await client.post("/auth/register", json=payload)

        assert response.status_code == 422

    async def test_returns_422_for_empty_name(self, client: AsyncClient, test_app: FastAPI) -> None:
        """名前が空で 422 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        payload = {**_VALID_PAYLOAD, "name": ""}
        response = await client.post("/auth/register", json=payload)

        assert response.status_code == 422

    async def test_email_is_lowercased(self, client: AsyncClient, test_app: FastAPI) -> None:
        """メールアドレスが小文字に正規化されること。"""
        session = _make_session()
        _override_deps(test_app, session=session)

        payload = {**_VALID_PAYLOAD, "email": "User@EXAMPLE.COM"}
        response = await client.post("/auth/register", json=payload)

        assert response.json()["email"] == "user@example.com"

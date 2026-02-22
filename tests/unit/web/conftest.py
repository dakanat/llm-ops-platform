"""Shared fixtures for web route tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.api.middleware.auth import TokenPayload, create_access_token
from src.config import Settings


def _create_web_app() -> FastAPI:
    """Create a FastAPI app with web routes registered."""
    settings = Settings()
    from src.main import create_app

    return create_app(settings)


@pytest.fixture(scope="module")
def test_app() -> FastAPI:
    """Create a FastAPI app for testing."""
    return _create_web_app()


@pytest_asyncio.fixture(scope="module")
async def client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Module-scoped async HTTP test client."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_overrides(test_app: FastAPI) -> Iterator[None]:
    """Clear dependency overrides after each test."""
    yield
    test_app.dependency_overrides.clear()


@pytest.fixture
def admin_user() -> TokenPayload:
    """Return a TokenPayload with admin role."""
    return TokenPayload(
        sub="user-1",
        email="admin@example.com",
        role="admin",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.fixture
def user_role() -> TokenPayload:
    """Return a TokenPayload with user role."""
    return TokenPayload(
        sub="user-2",
        email="user@example.com",
        role="user",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.fixture
def viewer_user() -> TokenPayload:
    """Return a TokenPayload with viewer role."""
    return TokenPayload(
        sub="user-3",
        email="viewer@example.com",
        role="viewer",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.fixture
def admin_token() -> str:
    """Return a valid JWT token with admin role."""
    return create_access_token(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        email="admin@example.com",
        role="admin",
    )


@pytest.fixture
def user_token() -> str:
    """Return a valid JWT token with user role."""
    return create_access_token(
        user_id=UUID("00000000-0000-0000-0000-000000000002"),
        email="user@example.com",
        role="user",
    )


class AuthCookies:
    """Context manager: set auth cookie on client, then clear on exit."""

    def __init__(self, client: AsyncClient, token: str) -> None:
        self._client = client
        self._token = token

    def __enter__(self) -> None:
        self._client.cookies.set("access_token", self._token)

    def __exit__(self, *args: object) -> None:
        self._client.cookies.delete("access_token")

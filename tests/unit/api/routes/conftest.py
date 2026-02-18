"""Shared fixtures for API route tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from src.api.middleware.auth import TokenPayload
from src.main import app


@pytest_asyncio.fixture(scope="module")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Module-scoped async HTTP test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    """Clear dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


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
def viewer_user() -> TokenPayload:
    """Return a TokenPayload with viewer role."""
    return TokenPayload(
        sub="user-3",
        email="viewer@example.com",
        role="viewer",
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

"""Shared fixtures for middleware tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
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

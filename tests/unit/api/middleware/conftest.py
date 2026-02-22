"""Shared fixtures for middleware tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.config import Settings
from src.main import create_app


@pytest.fixture(scope="module")
def test_app() -> FastAPI:
    """Create a FastAPI app for testing."""
    settings = Settings()
    return create_app(settings)


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

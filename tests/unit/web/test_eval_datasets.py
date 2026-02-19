"""Tests for web Eval datasets routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.config import Settings
from src.main import create_app

from tests.unit.web.conftest import auth_cookies


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
    """Return a valid JWT token."""
    from src.api.middleware.auth import create_access_token

    return create_access_token(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        email="admin@example.com",
        role="admin",
    )


class TestEvalDatasetsPage:
    """Tests for GET /web/eval/datasets."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/eval/datasets")
        assert resp.status_code == 303

    async def test_authenticated_returns_dataset_list(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import src.web.routes.eval_datasets as ds_module

        original = ds_module._get_session

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.exec.return_value = mock_result

        async def _fake_session() -> AsyncMock:
            return mock_session

        ds_module._get_session = _fake_session

        try:
            resp = await client.get("/web/eval/datasets", cookies=auth_cookies(admin_token))
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
            assert "Dataset" in resp.text or "dataset" in resp.text
        finally:
            ds_module._get_session = original


class TestEvalDatasetCreate:
    """Tests for GET /web/eval/datasets/create."""

    async def test_create_form_renders(self, client: AsyncClient, admin_token: str) -> None:
        resp = await client.get("/web/eval/datasets/create", cookies=auth_cookies(admin_token))
        assert resp.status_code == 200
        assert "form" in resp.text.lower()

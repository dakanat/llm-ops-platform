"""Tests for web RAG routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.config import Settings
from src.llm.providers.base import TokenUsage
from src.main import create_app
from src.rag.generator import GenerationResult
from src.rag.retriever import RetrievedChunk

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
    """Return a valid JWT token."""
    from uuid import UUID

    from src.api.middleware.auth import create_access_token

    return create_access_token(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        email="admin@example.com",
        role="admin",
    )


class TestRagPage:
    """Tests for GET /web/rag."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/rag")
        assert resp.status_code == 303

    async def test_authenticated_returns_rag_page(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/rag")
        assert resp.status_code == 200
        assert "RAG" in resp.text


class TestRagQuery:
    """Tests for POST /web/rag/query."""

    async def test_query_returns_result_fragment(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        mock_pipeline = AsyncMock()
        mock_pipeline.query.return_value = GenerationResult(
            answer="This is the answer from RAG.",
            sources=[
                RetrievedChunk(
                    document_id=UUID("00000000-0000-0000-0000-000000000001"),
                    content="Source content here",
                    chunk_index=0,
                ),
            ],
            model="test-model",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )

        from src.api.dependencies import get_rag_pipeline

        async def _mock_pipeline() -> AsyncGenerator[AsyncMock, None]:
            yield mock_pipeline

        test_app.dependency_overrides[get_rag_pipeline] = _mock_pipeline

        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/rag/query",
                data={"query": "What is RAG?"},
            )
        assert resp.status_code == 200
        assert "This is the answer from RAG." in resp.text
        assert "Source content here" in resp.text

        test_app.dependency_overrides.clear()

    async def test_empty_query_returns_error(self, client: AsyncClient, admin_token: str) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/rag/query",
                data={"query": ""},
            )
        assert resp.status_code == 200
        body = resp.text
        assert "alert" in body.lower() or "error" in body.lower() or "empty" in body.lower()

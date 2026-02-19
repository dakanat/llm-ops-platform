"""Integration tests for API endpoints.

Tests FastAPI endpoints with real dependency wiring where possible.
Database sessions use real PostgreSQL+pgvector; LLM and embedding
services are mocked. Auth is overridden per test via dependency_overrides.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession
from src.api.dependencies import (
    get_cost_tracker,
    get_llm_model,
    get_llm_provider,
    get_rag_pipeline,
)
from src.api.middleware.auth import TokenPayload, get_current_user
from src.db.models import Document, User
from src.db.vector_store import VectorStore
from src.llm.providers.base import LLMResponse, TokenUsage
from src.monitoring.cost_tracker import CostTracker
from src.rag.chunker import RecursiveCharacterSplitter
from src.rag.generator import Generator
from src.rag.index_manager import IndexManager
from src.rag.pipeline import RAGPipeline
from src.rag.preprocessor import Preprocessor
from src.rag.retriever import Retriever

from .conftest import FakeEmbedder, make_mock_llm_provider

pytestmark = pytest.mark.asyncio(loop_scope="module")


# ===========================================================================
# TestHealthEndpoint
# ===========================================================================
class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    async def test_health_returns_200_ok(
        self,
        client: AsyncClient,
    ) -> None:
        """GET /health returns 200 with {"status": "ok"}."""
        response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ===========================================================================
# TestChatEndpointIntegration
# ===========================================================================
class TestChatEndpointIntegration:
    """Tests for the /chat endpoint with mock LLM provider."""

    async def test_chat_returns_llm_response(
        self,
        test_app: FastAPI,
        client: AsyncClient,
    ) -> None:
        """POST /chat returns 200 with content matching mock LLM."""
        provider = make_mock_llm_provider(content="Integration test response")
        test_app.dependency_overrides[get_llm_provider] = lambda: provider
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        response = await client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Integration test response"

    async def test_chat_streaming_returns_sse(
        self,
        test_app: FastAPI,
        client: AsyncClient,
    ) -> None:
        """POST /chat with stream=true returns SSE events."""
        provider = make_mock_llm_provider(content="Streamed chunk")
        test_app.dependency_overrides[get_llm_provider] = lambda: provider
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        response = await client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "data:" in body


# ===========================================================================
# TestRAGQueryEndpointIntegration
# ===========================================================================
class TestRAGQueryEndpointIntegration:
    """Tests for the /rag/query endpoint with real DB + FakeEmbedder."""

    def _make_rag_pipeline_override(
        self,
        db_session: SQLModelAsyncSession,
        llm_content: str = "RAG answer",
    ) -> RAGPipeline:
        """Build a RAGPipeline for dependency override."""
        embedder = FakeEmbedder()
        vector_store = VectorStore(session=db_session)
        preprocessor = Preprocessor()
        chunker = RecursiveCharacterSplitter(chunk_size=128, chunk_overlap=16)
        index_manager = IndexManager(
            preprocessor=preprocessor,
            chunker=chunker,
            embedder=embedder,  # type: ignore[arg-type]
            vector_store=vector_store,
        )
        retriever = Retriever(embedder=embedder, vector_store=vector_store)  # type: ignore[arg-type]
        provider = make_mock_llm_provider(content=llm_content)
        generator = Generator(llm_provider=provider, model="test-model")
        return RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=generator,
        )

    async def test_rag_query_returns_answer_from_indexed_chunks(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """POST /rag/query returns 200 with answer and sources from DB."""
        pipeline = self._make_rag_pipeline_override(db_session, llm_content="The answer from RAG")

        # Index a document
        doc = Document(
            title="RAG API Test",
            content="Machine learning is a subset of AI. " * 10,
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()
        await pipeline.index_document(doc)

        # Override dependencies
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        async def _override_pipeline() -> AsyncGenerator[RAGPipeline, None]:
            yield pipeline

        test_app.dependency_overrides[get_rag_pipeline] = _override_pipeline

        response = await client.post(
            "/rag/query",
            json={"query": "What is machine learning?", "top_k": 3},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The answer from RAG"
        assert len(data["sources"]) > 0

    async def test_rag_query_filters_by_document_id(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Sources only from the specified document are returned."""
        pipeline = self._make_rag_pipeline_override(db_session)

        doc_a = Document(
            title="Doc A", content="Alpha content for testing. " * 10, user_id=test_user.id
        )
        doc_b = Document(
            title="Doc B", content="Beta content for testing. " * 10, user_id=test_user.id
        )
        db_session.add_all([doc_a, doc_b])
        await db_session.flush()
        await pipeline.index_document(doc_a)
        await pipeline.index_document(doc_b)

        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        async def _override_pipeline() -> AsyncGenerator[RAGPipeline, None]:
            yield pipeline

        test_app.dependency_overrides[get_rag_pipeline] = _override_pipeline

        response = await client.post(
            "/rag/query",
            json={
                "query": "Tell me about this",
                "document_id": str(doc_a.id),
            },
        )

        assert response.status_code == 200
        data = response.json()
        for src in data["sources"]:
            assert src["document_id"] == str(doc_a.id)

    async def test_rag_query_returns_422_for_empty_query(
        self,
        test_app: FastAPI,
        client: AsyncClient,
    ) -> None:
        """Empty query string triggers 422 validation error."""
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        response = await client.post(
            "/rag/query",
            json={"query": ""},
        )

        assert response.status_code == 422


# ===========================================================================
# TestAgentRunEndpointIntegration
# ===========================================================================
class TestAgentRunEndpointIntegration:
    """Tests for the /agent/run endpoint with auth and real tools."""

    def _setup_agent_overrides(
        self,
        test_app: FastAPI,
        user: TokenPayload,
        llm_responses: list[LLMResponse] | None = None,
    ) -> None:
        """Set up dependency overrides for agent endpoint tests."""
        test_app.dependency_overrides[get_current_user] = lambda: user
        test_app.dependency_overrides[get_llm_model] = lambda: "test-model"

        provider = AsyncMock()
        if llm_responses:
            provider.complete.side_effect = llm_responses
        else:
            provider.complete.return_value = LLMResponse(
                content="Thought: Simple question\nFinal Answer: Hello!",
                model="test-model",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            )
        test_app.dependency_overrides[get_llm_provider] = lambda: provider

    async def test_agent_run_returns_200_for_admin(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        admin_user_token: TokenPayload,
    ) -> None:
        """Admin role gets 200 with answer and steps."""
        self._setup_agent_overrides(test_app, admin_user_token)

        response = await client.post(
            "/agent/run",
            json={"query": "Hello", "max_steps": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "steps" in data

    async def test_agent_run_returns_200_for_user_role(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        user_role_token: TokenPayload,
    ) -> None:
        """User role (has AGENT_RUN permission) gets 200."""
        self._setup_agent_overrides(test_app, user_role_token)

        response = await client.post(
            "/agent/run",
            json={"query": "Hello", "max_steps": 5},
        )

        assert response.status_code == 200

    async def test_agent_run_returns_403_for_viewer(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        viewer_user_token: TokenPayload,
    ) -> None:
        """Viewer role (no AGENT_RUN permission) gets 403."""
        test_app.dependency_overrides[get_current_user] = lambda: viewer_user_token

        response = await client.post(
            "/agent/run",
            json={"query": "Hello", "max_steps": 5},
        )

        assert response.status_code == 403

    async def test_agent_run_with_real_calculator_tool(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        admin_user_token: TokenPayload,
    ) -> None:
        """Calculator tool step is visible in the response."""
        self._setup_agent_overrides(
            test_app,
            admin_user_token,
            llm_responses=[
                LLMResponse(
                    content=("Thought: I need to calculate\nAction: calculator\nAction Input: 3+4"),
                    model="test-model",
                    usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
                ),
                LLMResponse(
                    content="Thought: Got the answer\nFinal Answer: 3+4=7",
                    model="test-model",
                    usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
                ),
            ],
        )

        response = await client.post(
            "/agent/run",
            json={"query": "What is 3+4?", "max_steps": 5},
        )

        assert response.status_code == 200
        data = response.json()
        # First step should be the calculator action
        calc_step = data["steps"][0]
        assert calc_step["action"] == "calculator"
        assert calc_step["observation"] == "7"


# ===========================================================================
# TestAdminEndpointsIntegration
# ===========================================================================
class TestAdminEndpointsIntegration:
    """Tests for /admin/metrics and /admin/costs endpoints."""

    async def test_admin_metrics_returns_200_for_admin(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        admin_user_token: TokenPayload,
    ) -> None:
        """Admin gets 200 with text/plain Prometheus metrics."""
        test_app.dependency_overrides[get_current_user] = lambda: admin_user_token

        response = await client.get("/admin/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    async def test_admin_metrics_returns_403_for_user(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        user_role_token: TokenPayload,
    ) -> None:
        """User role (no ADMIN_READ permission) gets 403."""
        test_app.dependency_overrides[get_current_user] = lambda: user_role_token

        response = await client.get("/admin/metrics")

        assert response.status_code == 403

    async def test_admin_costs_returns_report_for_admin(
        self,
        test_app: FastAPI,
        client: AsyncClient,
        admin_user_token: TokenPayload,
    ) -> None:
        """Admin gets 200 with JSON cost report containing total_cost."""
        test_app.dependency_overrides[get_current_user] = lambda: admin_user_token
        test_app.dependency_overrides[get_cost_tracker] = lambda: CostTracker(
            alert_threshold_daily_usd=100,
        )

        response = await client.get("/admin/costs")

        assert response.status_code == 200
        data = response.json()
        assert "total_cost" in data
        assert "model_costs" in data
        assert "alert_triggered" in data

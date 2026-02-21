"""Tests for web Eval routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.config import Settings
from src.eval.runner import EvalRunResult, ExampleResult, MetricSummary
from src.main import create_app

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


def _make_mock_session_with_datasets(
    datasets: list[tuple[MagicMock, int]],
) -> AsyncMock:
    """Create a mock session that returns given datasets with example counts."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = datasets
    mock_session.exec.return_value = mock_result
    return mock_session


class TestEvalPage:
    """Tests for GET /web/eval."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/eval")
        assert resp.status_code == 303

    async def test_authenticated_returns_eval_page(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        import src.web.routes.eval as eval_module

        original = eval_module._get_session
        mock_session = _make_mock_session_with_datasets([])
        eval_module._get_session = AsyncMock(return_value=mock_session)

        try:
            with AuthCookies(client, admin_token):
                resp = await client.get("/web/eval")
            assert resp.status_code == 200
            assert "Eval" in resp.text
        finally:
            eval_module._get_session = original

    async def test_eval_page_shows_dataset_select(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        """GET /web/eval should show a select dropdown with available datasets."""
        import src.web.routes.eval as eval_module
        from src.db.models import EvalDatasetRecord

        original = eval_module._get_session

        ds = MagicMock(spec=EvalDatasetRecord)
        ds.id = UUID("00000000-0000-0000-0000-000000000010")
        ds.name = "my-test-dataset"

        mock_session = _make_mock_session_with_datasets([(ds, 3)])
        eval_module._get_session = AsyncMock(return_value=mock_session)

        try:
            with AuthCookies(client, admin_token):
                resp = await client.get("/web/eval")
            assert resp.status_code == 200
            assert "my-test-dataset" in resp.text
            assert "select" in resp.text.lower()
        finally:
            eval_module._get_session = original

    async def test_eval_page_shows_no_datasets_message(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        """GET /web/eval with no datasets shows a helpful message."""
        import src.web.routes.eval as eval_module

        original = eval_module._get_session
        mock_session = _make_mock_session_with_datasets([])
        eval_module._get_session = AsyncMock(return_value=mock_session)

        try:
            with AuthCookies(client, admin_token):
                resp = await client.get("/web/eval")
            assert resp.status_code == 200
            # Should contain a link or message guiding user to create datasets
            assert "/web/eval/datasets" in resp.text
        finally:
            eval_module._get_session = original


class TestEvalRun:
    """Tests for POST /web/eval/run."""

    async def test_run_returns_result_fragment(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        import src.web.routes.eval as eval_module
        from src.api.dependencies import get_eval_runner

        original_session = eval_module._get_session
        original_run = eval_module._run_eval

        dataset_id = UUID("00000000-0000-0000-0000-000000000020")

        mock_runner = AsyncMock()
        mock_runner.run.return_value = EvalRunResult(
            dataset_name="test-dataset",
            results=[
                ExampleResult(
                    query="What is AI?",
                    rag_answer="AI stands for artificial intelligence.",
                    rag_context="AI is artificial intelligence.",
                    faithfulness_score=0.9,
                    relevance_score=0.85,
                    latency_seconds=0.5,
                ),
            ],
            faithfulness_summary=MetricSummary(mean=0.9, count=1),
            relevance_summary=MetricSummary(mean=0.85, count=1),
            latency_summary=MetricSummary(mean=0.5, count=1),
        )
        test_app.dependency_overrides[get_eval_runner] = lambda: mock_runner

        async def _mock_run(
            runner: object,
            session: object,
            dataset_id: UUID,
        ) -> EvalRunResult:
            result: EvalRunResult = await mock_runner.run(None)
            return result

        eval_module._run_eval = _mock_run

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/eval/run",
                    data={"dataset_id": str(dataset_id)},
                )
            assert resp.status_code == 200
            assert "test-dataset" in resp.text
            assert "0.9" in resp.text
        finally:
            eval_module._run_eval = original_run
            eval_module._get_session = original_session
            test_app.dependency_overrides.clear()

    async def test_run_with_missing_dataset_id_returns_error(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        """POST /web/eval/run without dataset_id returns an error."""
        with AuthCookies(client, admin_token):
            resp = await client.post("/web/eval/run", data={})
        assert resp.status_code == 200
        assert "select" in resp.text.lower() or "required" in resp.text.lower()

    async def test_run_with_nonexistent_dataset_returns_error(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        """POST /web/eval/run with nonexistent dataset_id returns an error."""
        import src.web.routes.eval as eval_module

        original_session = eval_module._get_session

        mock_session = AsyncMock()
        mock_session.get.return_value = None
        eval_module._get_session = AsyncMock(return_value=mock_session)

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/eval/run",
                    data={"dataset_id": "00000000-0000-0000-0000-000000000099"},
                )
            assert resp.status_code == 200
            assert "not found" in resp.text.lower() or "error" in resp.text.lower()
        finally:
            eval_module._get_session = original_session

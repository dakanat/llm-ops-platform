"""Tests for web Eval routes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.config import Settings
from src.eval.datasets import EvalDataset, EvalExample
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


class TestEvalPage:
    """Tests for GET /web/eval."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        resp = await client.get("/web/eval")
        assert resp.status_code == 303

    async def test_authenticated_returns_eval_page(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/eval")
        assert resp.status_code == 200
        assert "Eval" in resp.text


class TestEvalRun:
    """Tests for POST /web/eval/run."""

    async def test_run_returns_result_fragment(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        example = EvalExample(
            query="What is AI?",
            context="AI is artificial intelligence.",
            answer="AI stands for artificial intelligence.",
        )
        mock_runner = AsyncMock()
        mock_runner.run.return_value = EvalRunResult(
            dataset_name="test-dataset",
            results=[
                ExampleResult(
                    example=example,
                    faithfulness_score=0.9,
                    relevance_score=0.85,
                    latency_seconds=0.5,
                ),
            ],
            faithfulness_summary=MetricSummary(mean=0.9, count=1),
            relevance_summary=MetricSummary(mean=0.85, count=1),
            latency_summary=MetricSummary(mean=0.5, count=1),
        )

        from src.api.dependencies import get_eval_runner

        test_app.dependency_overrides[get_eval_runner] = lambda: mock_runner

        import src.web.routes.eval as eval_module

        original_run = eval_module._run_eval

        async def _mock_run(
            runner: object, dataset_name: str, examples_data: list[dict[str, str]]
        ) -> EvalRunResult:
            dataset = EvalDataset(
                name=dataset_name,
                examples=[EvalExample(**ex) for ex in examples_data],
            )
            result: EvalRunResult = await mock_runner.run(dataset)
            return result

        eval_module._run_eval = _mock_run

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/eval/run",
                    data={
                        "dataset_name": "test-dataset",
                        "examples": (
                            '[{"query":"What is AI?",'
                            '"context":"AI is artificial intelligence.",'
                            '"answer":"AI stands for artificial intelligence."}]'
                        ),
                    },
                )
            assert resp.status_code == 200
            assert "test-dataset" in resp.text
            assert "0.9" in resp.text
        finally:
            eval_module._run_eval = original_run
            test_app.dependency_overrides.clear()

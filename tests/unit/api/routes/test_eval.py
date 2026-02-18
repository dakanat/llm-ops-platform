"""Tests for POST /eval/run endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

from httpx import AsyncClient
from src.api.middleware.auth import TokenPayload
from src.eval import EvalError
from src.eval.datasets import EvalExample
from src.eval.runner import EvalRunResult, ExampleResult, MetricSummary
from src.main import app


def _make_eval_run_result(
    dataset_name: str = "test-dataset",
    faithfulness_summary: MetricSummary | None = None,
    relevance_summary: MetricSummary | None = None,
) -> EvalRunResult:
    """Create a test EvalRunResult."""
    example = EvalExample(
        query="What is RAG?",
        context="RAG stands for Retrieval-Augmented Generation.",
        answer="RAG is Retrieval-Augmented Generation.",
        expected_answer="Retrieval-Augmented Generation",
    )
    return EvalRunResult(
        dataset_name=dataset_name,
        results=[
            ExampleResult(
                example=example,
                faithfulness_score=0.9,
                relevance_score=0.85,
                latency_seconds=0.5,
            ),
        ],
        faithfulness_summary=faithfulness_summary or MetricSummary(mean=0.9, count=1),
        relevance_summary=relevance_summary or MetricSummary(mean=0.85, count=1),
    )


def _mock_eval_runner(result: EvalRunResult | None = None) -> AsyncMock:
    """Create a mock EvalRunner."""
    runner = AsyncMock()
    runner.run.return_value = result or _make_eval_run_result()
    return runner


def _override_dependencies(
    runner: AsyncMock | None = None,
    user: TokenPayload | None = None,
    admin_user: TokenPayload | None = None,
) -> None:
    """Set FastAPI dependency overrides for eval route tests."""
    from src.api.dependencies import get_eval_runner
    from src.api.middleware.auth import get_current_user

    mock_runner = runner or _mock_eval_runner()

    app.dependency_overrides[get_eval_runner] = lambda: mock_runner
    app.dependency_overrides[get_current_user] = lambda: user or admin_user


_VALID_EXAMPLES = [
    {
        "query": "What is RAG?",
        "context": "RAG stands for Retrieval-Augmented Generation.",
        "answer": "RAG is Retrieval-Augmented Generation.",
        "expected_answer": "Retrieval-Augmented Generation",
    },
]


class TestEvalRunRoute:
    """POST /eval/run のテスト。"""

    async def test_returns_200_with_valid_dataset(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """有効なデータセットで 200 が返ること。"""
        _override_dependencies(admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 200

    async def test_returns_dataset_name(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """dataset_name がレスポンスに含まれること。"""
        result = _make_eval_run_result(dataset_name="my-dataset")
        runner = _mock_eval_runner(result=result)
        _override_dependencies(runner=runner, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "my-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.json()["dataset_name"] == "my-dataset"

    async def test_returns_results_with_scores(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """各結果にスコアが含まれること。"""
        _override_dependencies(admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["faithfulness_score"] == 0.9
        assert results[0]["relevance_score"] == 0.85

    async def test_returns_faithfulness_summary(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """faithfulness_summary がレスポンスに含まれること。"""
        result = _make_eval_run_result(
            faithfulness_summary=MetricSummary(mean=0.95, count=2),
        )
        _override_dependencies(runner=_mock_eval_runner(result=result), admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        summary = response.json()["faithfulness_summary"]
        assert summary["mean"] == 0.95
        assert summary["count"] == 2

    async def test_returns_relevance_summary(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """relevance_summary がレスポンスに含まれること。"""
        result = _make_eval_run_result(
            relevance_summary=MetricSummary(mean=0.8, count=3),
        )
        _override_dependencies(runner=_mock_eval_runner(result=result), admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        summary = response.json()["relevance_summary"]
        assert summary["mean"] == 0.8
        assert summary["count"] == 3

    async def test_passes_dataset_to_runner(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """データセットが EvalRunner.run() に渡されること。"""
        runner = _mock_eval_runner()
        _override_dependencies(runner=runner, admin_user=admin_user)
        await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        runner.run.assert_called_once()
        dataset = runner.run.call_args[0][0]
        assert dataset.name == "test-dataset"
        assert len(dataset.examples) == 1
        assert dataset.examples[0].query == "What is RAG?"

    async def test_returns_422_for_missing_dataset_name(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """dataset_name がない場合に 422 が返ること。"""
        _override_dependencies(admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 422

    async def test_returns_422_for_missing_examples(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """examples がない場合に 422 が返ること。"""
        _override_dependencies(admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset"},
        )

        assert response.status_code == 422

    async def test_returns_422_for_empty_examples(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """空の examples で 422 が返ること。"""
        _override_dependencies(admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": []},
        )

        assert response.status_code == 422

    async def test_returns_502_when_eval_fails(
        self, client: AsyncClient, admin_user: TokenPayload
    ) -> None:
        """EvalError 発生時に 502 が返ること。"""
        runner = _mock_eval_runner()
        runner.run.side_effect = EvalError("Evaluation failed")
        _override_dependencies(runner=runner, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 502

    async def test_returns_401_without_auth(self, client: AsyncClient) -> None:
        """認証なしで 401 が返ること。"""
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code in (401, 403)

    async def test_returns_403_for_viewer_role(
        self, client: AsyncClient, viewer_user: TokenPayload
    ) -> None:
        """viewer ロールで 403 が返ること。"""
        _override_dependencies(user=viewer_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 403

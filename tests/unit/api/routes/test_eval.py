"""Tests for POST /eval/run endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import AsyncClient
from src.api.middleware.auth import TokenPayload
from src.eval import EvalError
from src.eval.runner import EvalRunResult, ExampleResult, MetricSummary


def _make_eval_run_result(
    dataset_name: str = "test-dataset",
    faithfulness_summary: MetricSummary | None = None,
    relevance_summary: MetricSummary | None = None,
) -> EvalRunResult:
    """Create a test EvalRunResult."""
    return EvalRunResult(
        dataset_name=dataset_name,
        results=[
            ExampleResult(
                query="What is RAG?",
                expected_answer="Retrieval-Augmented Generation",
                rag_answer="RAG is Retrieval-Augmented Generation.",
                rag_context="RAG stands for Retrieval-Augmented Generation.",
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
    app: FastAPI,
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
        "expected_answer": "Retrieval-Augmented Generation",
    },
]


class TestEvalRunRoute:
    """POST /eval/run のテスト。"""

    async def test_returns_200_with_valid_dataset(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """有効なデータセットで 200 が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 200

    async def test_returns_dataset_name(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """dataset_name がレスポンスに含まれること。"""
        result = _make_eval_run_result(dataset_name="my-dataset")
        runner = _mock_eval_runner(result=result)
        _override_dependencies(test_app, runner=runner, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "my-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.json()["dataset_name"] == "my-dataset"

    async def test_returns_results_with_scores(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """各結果にスコアが含まれること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["faithfulness_score"] == 0.9
        assert results[0]["relevance_score"] == 0.85

    async def test_returns_faithfulness_summary(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """faithfulness_summary がレスポンスに含まれること。"""
        result = _make_eval_run_result(
            faithfulness_summary=MetricSummary(mean=0.95, count=2),
        )
        _override_dependencies(
            test_app, runner=_mock_eval_runner(result=result), admin_user=admin_user
        )
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        summary = response.json()["faithfulness_summary"]
        assert summary["mean"] == 0.95
        assert summary["count"] == 2

    async def test_returns_relevance_summary(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """relevance_summary がレスポンスに含まれること。"""
        result = _make_eval_run_result(
            relevance_summary=MetricSummary(mean=0.8, count=3),
        )
        _override_dependencies(
            test_app, runner=_mock_eval_runner(result=result), admin_user=admin_user
        )
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        summary = response.json()["relevance_summary"]
        assert summary["mean"] == 0.8
        assert summary["count"] == 3

    async def test_passes_dataset_to_runner(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """データセットが EvalRunner.run() に渡されること。"""
        runner = _mock_eval_runner()
        _override_dependencies(test_app, runner=runner, admin_user=admin_user)
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
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """dataset_name がない場合に 422 が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 422

    async def test_returns_422_for_missing_examples(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """examples がない場合に 422 が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset"},
        )

        assert response.status_code == 422

    async def test_returns_422_for_empty_examples(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """空の examples で 422 が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": []},
        )

        assert response.status_code == 422

    async def test_returns_502_when_eval_fails(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """EvalError 発生時に 502 が返ること。"""
        runner = _mock_eval_runner()
        runner.run.side_effect = EvalError("Evaluation failed")
        _override_dependencies(test_app, runner=runner, admin_user=admin_user)
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
        self, client: AsyncClient, viewer_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """viewer ロールで 403 が返ること。"""
        _override_dependencies(test_app, user=viewer_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "test-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 403


class TestEvalRunWithDatasetId:
    """POST /eval/run の dataset_id パスのテスト。"""

    async def test_returns_200_with_dataset_id(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """dataset_id 指定で 200 が返ること。"""
        import uuid
        from unittest.mock import MagicMock

        from src.db.models import EvalDatasetRecord, EvalExampleRecord
        from src.db.session import get_session

        ds_id = uuid.uuid4()
        ds = EvalDatasetRecord(id=ds_id, name="saved-dataset", created_by=uuid.uuid4())
        example = EvalExampleRecord(
            dataset_id=ds_id,
            query="What is RAG?",
            expected_answer="Retrieval-Augmented Generation",
        )

        session = AsyncMock()
        session.get.return_value = ds
        result_proxy = MagicMock()
        result_proxy.all.return_value = [example]
        session.exec.return_value = result_proxy

        runner = _mock_eval_runner()
        _override_dependencies(test_app, runner=runner, admin_user=admin_user)
        test_app.dependency_overrides[get_session] = lambda: session

        response = await client.post(
            "/eval/run",
            json={"dataset_name": "saved-dataset", "dataset_id": str(ds_id)},
        )

        assert response.status_code == 200

    async def test_loads_examples_from_db(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """dataset_id から DB の examples がロードされること。"""
        import uuid
        from unittest.mock import MagicMock

        from src.db.models import EvalDatasetRecord, EvalExampleRecord
        from src.db.session import get_session

        ds_id = uuid.uuid4()
        ds = EvalDatasetRecord(id=ds_id, name="saved-dataset", created_by=uuid.uuid4())
        example = EvalExampleRecord(
            dataset_id=ds_id,
            query="What is RAG?",
            expected_answer="Retrieval-Augmented Generation",
        )

        session = AsyncMock()
        session.get.return_value = ds
        result_proxy = MagicMock()
        result_proxy.all.return_value = [example]
        session.exec.return_value = result_proxy

        runner = _mock_eval_runner()
        _override_dependencies(test_app, runner=runner, admin_user=admin_user)
        test_app.dependency_overrides[get_session] = lambda: session

        await client.post(
            "/eval/run",
            json={"dataset_name": "saved-dataset", "dataset_id": str(ds_id)},
        )

        runner.run.assert_called_once()
        dataset = runner.run.call_args[0][0]
        assert dataset.name == "saved-dataset"
        assert len(dataset.examples) == 1
        assert dataset.examples[0].query == "What is RAG?"

    async def test_returns_404_for_nonexistent_dataset_id(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """存在しない dataset_id で 404 が返ること。"""
        import uuid

        from src.db.session import get_session

        session = AsyncMock()
        session.get.return_value = None

        runner = _mock_eval_runner()
        _override_dependencies(test_app, runner=runner, admin_user=admin_user)
        test_app.dependency_overrides[get_session] = lambda: session

        response = await client.post(
            "/eval/run",
            json={
                "dataset_name": "ds",
                "dataset_id": str(uuid.uuid4()),
            },
        )

        assert response.status_code == 404

    async def test_returns_422_when_both_dataset_id_and_examples(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """dataset_id と examples の両方指定で 422 が返ること。"""
        import uuid

        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={
                "dataset_name": "ds",
                "dataset_id": str(uuid.uuid4()),
                "examples": _VALID_EXAMPLES,
            },
        )

        assert response.status_code == 422

    async def test_returns_422_when_neither_dataset_id_nor_examples(
        self, client: AsyncClient, admin_user: TokenPayload, test_app: FastAPI
    ) -> None:
        """dataset_id も examples もない場合に 422 が返ること。"""
        _override_dependencies(test_app, admin_user=admin_user)
        response = await client.post(
            "/eval/run",
            json={"dataset_name": "ds"},
        )

        assert response.status_code == 422

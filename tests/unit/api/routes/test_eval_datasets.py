"""Tests for eval dataset CRUD and synthetic generation endpoints."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from httpx import AsyncClient
from src.api.middleware.auth import TokenPayload
from src.eval import SyntheticDataError
from src.eval.datasets import EvalDataset, EvalExample

_USER_UUID = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))

_VALID_EXAMPLES = [
    {
        "query": "What is RAG?",
        "context": "RAG stands for Retrieval-Augmented Generation.",
        "answer": "RAG is Retrieval-Augmented Generation.",
        "expected_answer": "Retrieval-Augmented Generation",
    },
]


def _make_admin_user() -> TokenPayload:
    """Create an admin TokenPayload with a valid UUID sub."""
    from datetime import UTC, datetime, timedelta

    return TokenPayload(
        sub=_USER_UUID,
        email="admin@example.com",
        role="admin",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


def _make_viewer_user() -> TokenPayload:
    """Create a viewer TokenPayload."""
    from datetime import UTC, datetime, timedelta

    return TokenPayload(
        sub=str(uuid.UUID("00000000-0000-0000-0000-000000000003")),
        email="viewer@example.com",
        role="viewer",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


def _override_deps(
    app: FastAPI,
    *,
    session: AsyncMock | None = None,
    user: TokenPayload | None = None,
    generator: AsyncMock | None = None,
) -> None:
    """Set dependency overrides for eval dataset tests."""
    from src.api.dependencies import get_synthetic_data_generator
    from src.api.middleware.auth import get_current_user
    from src.db.session import get_session

    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    if session is not None:
        app.dependency_overrides[get_session] = lambda: session
    if generator is not None:
        app.dependency_overrides[get_synthetic_data_generator] = lambda: generator


def _make_session(
    *,
    exec_result: list[object] | None = None,
    get_result: object | None = None,
) -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()

    # exec returns a result proxy that has .all() and .first()
    result_proxy = MagicMock()
    result_proxy.all.return_value = exec_result or []
    result_proxy.first.return_value = exec_result[0] if exec_result else None

    session.exec.return_value = result_proxy
    session.get.return_value = get_result

    return session


class TestCreateEvalDataset:
    """POST /eval/datasets のテスト。"""

    async def test_returns_201_on_success(self, client: AsyncClient, test_app: FastAPI) -> None:
        """正常系で 201 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.post(
            "/eval/datasets",
            json={"name": "my-dataset", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 201

    async def test_response_contains_dataset_fields(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """レスポンスに必要なフィールドが含まれること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.post(
            "/eval/datasets",
            json={"name": "my-dataset", "examples": _VALID_EXAMPLES},
        )

        data = response.json()
        assert data["name"] == "my-dataset"
        assert "id" in data
        assert "examples" in data
        assert len(data["examples"]) == 1
        assert data["examples"][0]["query"] == "What is RAG?"

    async def test_response_includes_description(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """description が設定されたらレスポンスに含まれること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.post(
            "/eval/datasets",
            json={
                "name": "my-dataset",
                "description": "Test description",
                "examples": _VALID_EXAMPLES,
            },
        )

        assert response.json()["description"] == "Test description"

    async def test_calls_session_add_and_commit(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """session.add と session.commit が呼ばれること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user())

        await client.post(
            "/eval/datasets",
            json={"name": "my-dataset", "examples": _VALID_EXAMPLES},
        )

        assert session.add.called
        assert session.commit.called

    async def test_returns_409_on_duplicate_name(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """重複 name で 409 が返ること。"""
        from sqlalchemy.exc import IntegrityError

        session = _make_session()
        session.commit.side_effect = IntegrityError("duplicate", params={}, orig=Exception())
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.post(
            "/eval/datasets",
            json={"name": "duplicate", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 409

    async def test_returns_422_for_empty_name(self, client: AsyncClient, test_app: FastAPI) -> None:
        """空の name で 422 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.post(
            "/eval/datasets",
            json={"name": "", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 422

    async def test_returns_422_for_empty_examples(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """空の examples で 422 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.post(
            "/eval/datasets",
            json={"name": "ds", "examples": []},
        )

        assert response.status_code == 422

    async def test_returns_401_without_auth(self, client: AsyncClient) -> None:
        """認証なしで 401 が返ること。"""
        response = await client.post(
            "/eval/datasets",
            json={"name": "ds", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code in (401, 403)

    async def test_returns_403_for_viewer_role(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """viewer ロールで 403 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_viewer_user())

        response = await client.post(
            "/eval/datasets",
            json={"name": "ds", "examples": _VALID_EXAMPLES},
        )

        assert response.status_code == 403


class TestListEvalDatasets:
    """GET /eval/datasets のテスト。"""

    async def test_returns_200(self, client: AsyncClient, test_app: FastAPI) -> None:
        """正常系で 200 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.get("/eval/datasets")

        assert response.status_code == 200

    async def test_returns_empty_list(self, client: AsyncClient, test_app: FastAPI) -> None:
        """データなしで空リストが返ること。"""
        session = _make_session(exec_result=[])
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.get("/eval/datasets")

        assert response.json() == []

    async def test_returns_dataset_list_with_example_count(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """データセット一覧に example_count が含まれること。"""
        from src.db.models import EvalDatasetRecord

        ds = EvalDatasetRecord(
            name="ds-1",
            created_by=uuid.UUID(_USER_UUID),
        )

        # The list endpoint uses a query that returns (dataset, count) tuples
        session = _make_session(exec_result=[(ds, 5)])
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.get("/eval/datasets")

        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "ds-1"
        assert data[0]["example_count"] == 5

    async def test_viewer_can_list(self, client: AsyncClient, test_app: FastAPI) -> None:
        """viewer ロールでも一覧取得できること (EVAL_READ 権限)。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_viewer_user())

        response = await client.get("/eval/datasets")

        assert response.status_code == 200


class TestGetEvalDataset:
    """GET /eval/datasets/{dataset_id} のテスト。"""

    async def test_returns_200_with_examples(self, client: AsyncClient, test_app: FastAPI) -> None:
        """正常系で 200 + examples が返ること。"""
        from src.db.models import EvalDatasetRecord, EvalExampleRecord

        ds_id = uuid.uuid4()
        ds = EvalDatasetRecord(id=ds_id, name="ds-1", created_by=uuid.uuid4())

        example = EvalExampleRecord(
            dataset_id=ds_id,
            query="What is RAG?",
            context="RAG stands for Retrieval-Augmented Generation.",
            answer="RAG is Retrieval-Augmented Generation.",
        )

        session = _make_session()
        session.get.return_value = ds
        # exec for examples query
        result_proxy = MagicMock()
        result_proxy.all.return_value = [example]
        session.exec.return_value = result_proxy

        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.get(f"/eval/datasets/{ds_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ds-1"
        assert len(data["examples"]) == 1
        assert data["examples"][0]["query"] == "What is RAG?"

    async def test_returns_404_for_nonexistent(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """存在しない ID で 404 が返ること。"""
        session = _make_session()
        session.get.return_value = None
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.get(f"/eval/datasets/{uuid.uuid4()}")

        assert response.status_code == 404


class TestDeleteEvalDataset:
    """DELETE /eval/datasets/{dataset_id} のテスト。"""

    async def test_returns_204_on_success(self, client: AsyncClient, test_app: FastAPI) -> None:
        """正常系で 204 が返ること。"""
        from src.db.models import EvalDatasetRecord

        ds = EvalDatasetRecord(name="ds-1", created_by=uuid.uuid4())
        session = _make_session()
        session.get.return_value = ds
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.delete(f"/eval/datasets/{ds.id}")

        assert response.status_code == 204

    async def test_calls_session_delete(self, client: AsyncClient, test_app: FastAPI) -> None:
        """session.delete が呼ばれること。"""
        from src.db.models import EvalDatasetRecord

        ds = EvalDatasetRecord(name="ds-1", created_by=uuid.uuid4())
        session = _make_session()
        session.get.return_value = ds
        _override_deps(test_app, session=session, user=_make_admin_user())

        await client.delete(f"/eval/datasets/{ds.id}")

        session.delete.assert_called_once_with(ds)

    async def test_returns_404_for_nonexistent(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """存在しない ID で 404 が返ること。"""
        session = _make_session()
        session.get.return_value = None
        _override_deps(test_app, session=session, user=_make_admin_user())

        response = await client.delete(f"/eval/datasets/{uuid.uuid4()}")

        assert response.status_code == 404

    async def test_returns_403_for_viewer(self, client: AsyncClient, test_app: FastAPI) -> None:
        """viewer ロールで 403 が返ること。"""
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_viewer_user())

        response = await client.delete(f"/eval/datasets/{uuid.uuid4()}")

        assert response.status_code == 403


class TestGenerateSyntheticDataset:
    """POST /eval/datasets/generate のテスト。"""

    async def test_returns_201_on_success(self, client: AsyncClient, test_app: FastAPI) -> None:
        """正常系で 201 が返ること。"""
        generator = AsyncMock()
        generator.generate.return_value = EvalDataset(
            name="synthetic",
            examples=[
                EvalExample(
                    query="Q1",
                    context="source text",
                    answer="A1",
                    expected_answer="A1",
                ),
            ],
        )
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user(), generator=generator)

        response = await client.post(
            "/eval/datasets/generate",
            json={"name": "gen-dataset", "text": "Some document text"},
        )

        assert response.status_code == 201

    async def test_response_contains_generated_examples(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """レスポンスに生成された examples が含まれること。"""
        generator = AsyncMock()
        generator.generate.return_value = EvalDataset(
            name="synthetic",
            examples=[
                EvalExample(
                    query="Q1",
                    context="source text",
                    answer="A1",
                    expected_answer="A1",
                ),
            ],
        )
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user(), generator=generator)

        response = await client.post(
            "/eval/datasets/generate",
            json={"name": "gen-dataset", "text": "Some document text"},
        )

        data = response.json()
        assert data["name"] == "gen-dataset"
        assert len(data["examples"]) == 1
        assert data["examples"][0]["query"] == "Q1"

    async def test_passes_num_pairs_to_generator(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """num_pairs が generator に渡されること。"""
        generator = AsyncMock()
        generator.generate.return_value = EvalDataset(
            name="synthetic",
            examples=[
                EvalExample(query="Q1", context="c", answer="A1", expected_answer="A1"),
            ],
        )
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user(), generator=generator)

        await client.post(
            "/eval/datasets/generate",
            json={"name": "gen-dataset", "text": "Some text", "num_pairs": 5},
        )

        generator.generate.assert_called_once()
        call_kwargs = generator.generate.call_args
        assert call_kwargs.kwargs.get("num_pairs") == 5 or call_kwargs[1].get("num_pairs") == 5

    async def test_returns_502_on_generation_failure(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """生成失敗時に 502 が返ること。"""
        generator = AsyncMock()
        generator.generate.side_effect = SyntheticDataError("LLM failed")
        session = _make_session()
        _override_deps(test_app, session=session, user=_make_admin_user(), generator=generator)

        response = await client.post(
            "/eval/datasets/generate",
            json={"name": "gen-dataset", "text": "Some text"},
        )

        assert response.status_code == 502

    async def test_returns_409_on_duplicate_name(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """重複 name で 409 が返ること。"""
        from sqlalchemy.exc import IntegrityError

        generator = AsyncMock()
        generator.generate.return_value = EvalDataset(
            name="synthetic",
            examples=[
                EvalExample(query="Q1", context="c", answer="A1", expected_answer="A1"),
            ],
        )
        session = _make_session()
        session.commit.side_effect = IntegrityError("duplicate", params={}, orig=Exception())
        _override_deps(test_app, session=session, user=_make_admin_user(), generator=generator)

        response = await client.post(
            "/eval/datasets/generate",
            json={"name": "duplicate", "text": "Some text"},
        )

        assert response.status_code == 409

    async def test_returns_422_for_empty_text(self, client: AsyncClient, test_app: FastAPI) -> None:
        """空の text で 422 が返ること。"""
        session = _make_session()
        generator = AsyncMock()
        _override_deps(test_app, session=session, user=_make_admin_user(), generator=generator)

        response = await client.post(
            "/eval/datasets/generate",
            json={"name": "gen-dataset", "text": ""},
        )

        assert response.status_code == 422

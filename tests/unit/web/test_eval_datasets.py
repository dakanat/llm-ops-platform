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
            with AuthCookies(client, admin_token):
                resp = await client.get("/web/eval/datasets")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
            assert "Dataset" in resp.text or "dataset" in resp.text
        finally:
            ds_module._get_session = original


class TestEvalDatasetCreate:
    """Tests for GET /web/eval/datasets/create."""

    async def test_create_form_renders(self, client: AsyncClient, admin_token: str) -> None:
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/eval/datasets/create")
        assert resp.status_code == 200
        assert "form" in resp.text.lower()

    async def test_create_form_contains_generate_tab(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        """GET /web/eval/datasets/create should contain a Generate tab."""
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/eval/datasets/create")
        assert resp.status_code == 200
        assert "Generate" in resp.text


class TestEvalDatasetGenerate:
    """Tests for POST /web/eval/datasets/generate."""

    async def test_generate_success_redirects_to_detail(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        """Successful generation redirects to dataset detail page via HX-Redirect."""
        import src.web.routes.eval_datasets as ds_module
        from src.api.dependencies import get_synthetic_data_generator
        from src.eval.datasets import EvalDataset, EvalExample

        original = ds_module._get_session

        dataset_id = UUID("00000000-0000-0000-0000-000000000099")
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        async def _fake_session() -> AsyncMock:
            return mock_session

        ds_module._get_session = _fake_session

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = EvalDataset(
            name="synthetic",
            examples=[
                EvalExample(
                    query="What is AI?",
                    expected_answer="Artificial intelligence.",
                ),
            ],
        )
        test_app.dependency_overrides[get_synthetic_data_generator] = lambda: mock_generator

        original_helper = ds_module._generate_and_save

        async def _mock_generate_and_save(
            generator: object,
            session: object,
            name: str,
            description: str,
            text: str,
            num_pairs: int,
            created_by: UUID,
        ) -> MagicMock:
            from src.db.models import EvalDatasetRecord

            record = MagicMock(spec=EvalDatasetRecord)
            record.id = dataset_id
            return record

        ds_module._generate_and_save = _mock_generate_and_save

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/eval/datasets/generate",
                    data={
                        "name": "test-dataset",
                        "description": "A test dataset",
                        "text": "Some source text for generating QA pairs.",
                        "num_pairs": "5",
                    },
                )
            assert resp.status_code == 200
            assert resp.headers.get("hx-redirect") == f"/web/eval/datasets/{dataset_id}"
        finally:
            ds_module._get_session = original
            ds_module._generate_and_save = original_helper
            test_app.dependency_overrides.clear()

    async def test_generate_shows_error_on_empty_name(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        """Empty name returns error toast."""
        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/eval/datasets/generate",
                data={"name": "", "text": "Some text"},
            )
        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "required" in resp.text.lower()

    async def test_generate_shows_error_on_empty_text(
        self, client: AsyncClient, admin_token: str
    ) -> None:
        """Empty text returns error toast."""
        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/eval/datasets/generate",
                data={"name": "my-dataset", "text": ""},
            )
        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "required" in resp.text.lower()

    async def test_generate_shows_error_on_llm_failure(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        """SyntheticDataError returns error toast."""
        import src.web.routes.eval_datasets as ds_module
        from src.eval import SyntheticDataError

        original = ds_module._get_session
        mock_session = AsyncMock()

        async def _fake_session() -> AsyncMock:
            return mock_session

        ds_module._get_session = _fake_session

        original_helper = ds_module._generate_and_save

        async def _mock_generate_fail(*args: object, **kwargs: object) -> None:
            raise SyntheticDataError("LLM call failed")

        ds_module._generate_and_save = _mock_generate_fail  # type: ignore[assignment]

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/eval/datasets/generate",
                    data={"name": "fail-dataset", "text": "Some text"},
                )
            assert resp.status_code == 200
            assert "LLM call failed" in resp.text
        finally:
            ds_module._get_session = original
            ds_module._generate_and_save = original_helper
            test_app.dependency_overrides.clear()

    async def test_generate_shows_error_on_duplicate_name(
        self, test_app: FastAPI, client: AsyncClient, admin_token: str
    ) -> None:
        """IntegrityError returns error toast about duplicate name."""
        import src.web.routes.eval_datasets as ds_module

        original = ds_module._get_session
        mock_session = AsyncMock()

        async def _fake_session() -> AsyncMock:
            return mock_session

        ds_module._get_session = _fake_session

        original_helper = ds_module._generate_and_save

        async def _mock_generate_integrity(*args: object, **kwargs: object) -> None:
            from sqlalchemy.exc import IntegrityError

            raise IntegrityError("duplicate", params=None, orig=Exception("unique"))

        ds_module._generate_and_save = _mock_generate_integrity  # type: ignore[assignment]

        try:
            with AuthCookies(client, admin_token):
                resp = await client.post(
                    "/web/eval/datasets/generate",
                    data={"name": "dup-dataset", "text": "Some text"},
                )
            assert resp.status_code == 200
            assert "already exists" in resp.text.lower() or "duplicate" in resp.text.lower()
        finally:
            ds_module._get_session = original
            ds_module._generate_and_save = original_helper
            test_app.dependency_overrides.clear()

    async def test_generate_unauthenticated_redirects(self, client: AsyncClient) -> None:
        """Unauthenticated request redirects to login."""
        resp = await client.post(
            "/web/eval/datasets/generate",
            data={"name": "test", "text": "Some text"},
        )
        assert resp.status_code == 303

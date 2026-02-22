"""Shared fixtures for integration tests.

Provides real PostgreSQL+pgvector database fixtures with per-test
transaction rollback, a deterministic FakeEmbedder, test data factories,
mock LLM providers, and FastAPI test client fixtures.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import AsyncGenerator, Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession
from src.api.middleware.auth import TokenPayload, hash_password
from src.config import Settings
from src.db.models import Document, User
from src.db.vector_store import EMBEDDING_DIMENSIONS
from src.llm.providers.base import LLMChunk, LLMResponse, TokenUsage
from src.main import create_app

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_platform_test",
)

# Raw asyncpg URL for creating the database (no driver prefix)
_RAW_URL = TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


# ---------------------------------------------------------------------------
# Database engine (module-scoped, shared event loop)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def _engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create test database, enable pgvector, create tables, yield engine."""
    try:
        import asyncpg
    except ImportError:
        pytest.skip("asyncpg not installed")

    # Create database if needed
    try:
        base_url = _RAW_URL.rsplit("/", 1)[0] + "/postgres"
        conn = await asyncpg.connect(base_url)
        try:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = 'llm_platform_test'"
            )
            if not exists:
                await conn.execute("CREATE DATABASE llm_platform_test")
        finally:
            await conn.close()
    except (OSError, asyncpg.PostgresError):
        pytest.skip("PostgreSQL is not available — skipping integration tests")

    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        connect_args={
            "server_settings": {"timezone": "UTC"},
        },
    )

    # Enable pgvector + create tables
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Per-test database session with transaction rollback
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(loop_scope="module")
async def db_session(
    _engine: AsyncEngine,
) -> AsyncGenerator[SQLModelAsyncSession, None]:
    """Yield a session bound to an uncommitted transaction; rolls back after test."""
    async with _engine.connect() as conn:
        trans = await conn.begin()
        session = SQLModelAsyncSession(bind=conn)
        try:
            yield session
        finally:
            await trans.rollback()


# ---------------------------------------------------------------------------
# FakeEmbedder — deterministic, no GPU/network required
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic embedder that produces reproducible 1024-dim unit vectors.

    Uses ``hash(text)`` as seed for ``random.Random``. Accepts an explicit
    ``mappings`` dict to return specific vectors for controlled cosine
    similarity tests.
    """

    def __init__(self, mappings: dict[str, list[float]] | None = None) -> None:
        self._mappings = mappings or {}

    async def embed(self, text: str) -> list[float]:
        """Return a deterministic embedding for *text*."""
        if text in self._mappings:
            return self._mappings[text]
        return _make_deterministic_embedding(hash(text))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings for each text."""
        return [await self.embed(t) for t in texts]

    async def close(self) -> None:
        """No-op."""


def make_deterministic_embedding(seed: int) -> list[float]:
    """Create a deterministic 1024-dim unit vector from *seed*."""
    return _make_deterministic_embedding(seed)


def _make_deterministic_embedding(seed: int) -> list[float]:
    rng = random.Random(seed)  # noqa: S311
    raw = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIMENSIONS)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def make_similar_embedding(base: list[float], noise: float = 0.05) -> list[float]:
    """Return an embedding similar to *base* with small random noise."""
    rng = random.Random(42)  # noqa: S311
    noisy = [v + rng.gauss(0, noise) for v in base]
    norm = math.sqrt(sum(x * x for x in noisy))
    return [x / norm for x in noisy]


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(loop_scope="module")
async def test_user(db_session: SQLModelAsyncSession) -> User:
    """Create and flush an admin user."""
    user = User(
        email="integration-admin@example.com",
        name="Integration Admin",
        hashed_password=hash_password("test-password"),
        role="admin",
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture(loop_scope="module")
async def test_document(db_session: SQLModelAsyncSession, test_user: User) -> Document:
    """Create and flush a document linked to *test_user*."""
    doc = Document(
        title="Test Document",
        content="This is test content for integration testing.",
        user_id=test_user.id,
    )
    db_session.add(doc)
    await db_session.flush()
    return doc


# ---------------------------------------------------------------------------
# Mock LLM provider factory
# ---------------------------------------------------------------------------
def make_mock_llm_provider(
    content: str = "Mock LLM response",
    model: str = "test-model",
) -> AsyncMock:
    """Return an ``AsyncMock`` that mimics an ``LLMProvider``."""
    provider = AsyncMock()
    provider.complete.return_value = LLMResponse(
        content=content,
        model=model,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        finish_reason="stop",
    )

    async def _stream(*args: object, **kwargs: object) -> AsyncGenerator[LLMChunk, None]:
        yield LLMChunk(content=content, finish_reason="stop")

    provider.stream = _stream
    return provider


# ---------------------------------------------------------------------------
# FastAPI test app and HTTP client (module-scoped for API tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def test_app() -> FastAPI:
    """Create a FastAPI app for testing."""
    settings = Settings()
    return create_app(settings)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Module-scoped async HTTP test client."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_overrides(test_app: FastAPI) -> Iterator[None]:
    """Clear dependency overrides after each test."""
    yield
    test_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Auth fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def admin_user_token() -> TokenPayload:
    """TokenPayload with admin role."""
    return TokenPayload(
        sub="admin-1",
        email="admin@example.com",
        role="admin",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.fixture
def user_role_token() -> TokenPayload:
    """TokenPayload with user role."""
    return TokenPayload(
        sub="user-1",
        email="user@example.com",
        role="user",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )


@pytest.fixture
def viewer_user_token() -> TokenPayload:
    """TokenPayload with viewer role."""
    return TokenPayload(
        sub="viewer-1",
        email="viewer@example.com",
        role="viewer",
        exp=datetime.now(UTC) + timedelta(hours=1),
    )

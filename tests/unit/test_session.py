"""Tests for async database session management."""

import inspect


class TestGetSession:
    """get_session の基本検証。"""

    def test_get_session_is_async_generator(self) -> None:
        """get_session が async generator であること。"""
        from src.db.session import get_session

        assert inspect.isasyncgenfunction(get_session)

    def test_engine_uses_asyncpg_url(self) -> None:
        """engine が postgresql+asyncpg URL を使用していること。"""
        from src.db.session import engine

        assert "postgresql+asyncpg" in str(engine.url)

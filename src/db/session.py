"""Async database session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession

from src.config import Settings

settings = Settings()

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for FastAPI dependency injection."""
    async with SQLModelAsyncSession(engine) as session:
        yield session

"""FastAPI dependency injection providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.middleware.auth import TokenPayload, get_current_user
from src.config import Settings

if TYPE_CHECKING:
    from src.agent.tools.registry import ToolRegistry
    from src.eval.runner import EvalRunner
    from src.monitoring.cost_tracker import CostTracker
from src.db.session import get_session
from src.db.vector_store import VectorStore
from src.llm.providers.base import LLMProvider
from src.llm.router import LLMRouter
from src.rag.chunker import RecursiveCharacterSplitter
from src.rag.embedder import Embedder
from src.rag.generator import Generator
from src.rag.index_manager import IndexManager
from src.rag.pipeline import RAGPipeline
from src.rag.preprocessor import Preprocessor
from src.rag.retriever import Retriever

CurrentUser = Annotated[TokenPayload, Depends(get_current_user)]


def get_settings() -> Settings:
    """Return application settings."""
    return Settings()


def get_llm_router(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMRouter:
    """Return an LLM router configured from settings."""
    return LLMRouter(settings=settings)


def get_llm_provider(
    router: Annotated[LLMRouter, Depends(get_llm_router)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMProvider:
    """Return the configured LLM provider, wrapped with PII sanitizer if enabled."""
    from src.llm.pii_sanitizing_provider import PIISanitizingProvider

    provider = router.get_provider()
    if settings.pii_detection_enabled and settings.pii_mask_llm_outbound:
        return PIISanitizingProvider(inner=provider)
    return provider


def get_llm_model(
    router: Annotated[LLMRouter, Depends(get_llm_router)],
) -> str:
    """Return the configured LLM model name."""
    return router.model


async def get_rag_pipeline(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[RAGPipeline, None]:
    """Build a RAG pipeline and clean up the embedder on teardown."""
    embedder = Embedder(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )
    try:
        vector_store = VectorStore(session=session)
        preprocessor = Preprocessor()
        chunker = RecursiveCharacterSplitter()
        index_manager = IndexManager(
            preprocessor=preprocessor,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )
        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        provider: LLMProvider = LLMRouter(settings=settings).get_provider()
        if settings.pii_detection_enabled and settings.pii_mask_llm_outbound:
            from src.llm.pii_sanitizing_provider import PIISanitizingProvider

            provider = PIISanitizingProvider(inner=provider)
        generator = Generator(
            llm_provider=provider,
            model=settings.llm_model,
        )
        yield RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=generator,
        )
    finally:
        await embedder.close()


def get_tool_registry() -> ToolRegistry:
    """Return a ToolRegistry with default tools registered."""
    from src.agent.tools.calculator import CalculatorTool
    from src.agent.tools.registry import ToolRegistry as _ToolRegistry

    registry = _ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def get_eval_runner() -> EvalRunner:
    """Return an EvalRunner (no metrics by default — they require LLM)."""
    from src.eval.runner import EvalRunner as _EvalRunner

    return _EvalRunner()


_cost_tracker: CostTracker | None = None


def get_cost_tracker(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CostTracker:
    """Return a module-level CostTracker singleton."""
    from src.monitoring.cost_tracker import CostTracker as _CostTracker

    global _cost_tracker  # noqa: PLW0603
    if _cost_tracker is None:
        _cost_tracker = _CostTracker(
            alert_threshold_daily_usd=settings.cost_alert_threshold_daily_usd,
        )
    return _cost_tracker

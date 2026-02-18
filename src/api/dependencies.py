"""FastAPI dependency injection providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from src.config import Settings
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
) -> LLMProvider:
    """Return the configured LLM provider."""
    return router.get_provider()


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
        provider = LLMRouter(settings=settings).get_provider()
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

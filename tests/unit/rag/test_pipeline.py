"""Tests for RAGPipeline (index_document, query)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.rag.generator import GenerationError, GenerationResult, Generator
from src.rag.index_manager import IndexingError, IndexManager
from src.rag.pipeline import RAGPipeline, RAGPipelineError
from src.rag.retriever import RetrievalError, Retriever

from .conftest import make_chunk, make_document, make_retrieved_chunk


class TestRAGPipelineIndexDocument:
    """RAGPipeline.index_document() のテスト。"""

    @pytest.fixture
    def index_manager(self) -> AsyncMock:
        return AsyncMock(spec=IndexManager)

    @pytest.fixture
    def retriever(self) -> AsyncMock:
        return AsyncMock(spec=Retriever)

    @pytest.fixture
    def generator(self) -> AsyncMock:
        return AsyncMock(spec=Generator)

    @pytest.fixture
    def pipeline(
        self,
        index_manager: AsyncMock,
        retriever: AsyncMock,
        generator: AsyncMock,
    ) -> RAGPipeline:
        return RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=generator,
        )

    async def test_delegates_to_index_manager(
        self, pipeline: RAGPipeline, index_manager: AsyncMock
    ) -> None:
        """IndexManager.index_document に委譲されること。"""
        doc = make_document()
        expected_chunks = [make_chunk()]
        index_manager.index_document.return_value = expected_chunks

        result = await pipeline.index_document(doc)

        index_manager.index_document.assert_awaited_once_with(doc)
        assert result == expected_chunks

    async def test_wraps_indexing_error(
        self, pipeline: RAGPipeline, index_manager: AsyncMock
    ) -> None:
        """IndexingError が RAGPipelineError でラップされること。"""
        doc = make_document()
        indexing_error = IndexingError("index failed")
        index_manager.index_document.side_effect = indexing_error

        with pytest.raises(RAGPipelineError) as exc_info:
            await pipeline.index_document(doc)

        assert exc_info.value.__cause__ is indexing_error

    async def test_returns_chunks_from_index_manager(
        self, pipeline: RAGPipeline, index_manager: AsyncMock
    ) -> None:
        """IndexManager から返されたチャンクがそのまま返ること。"""
        doc = make_document()
        chunks = [make_chunk(), make_chunk(chunk_index=1)]
        index_manager.index_document.return_value = chunks

        result = await pipeline.index_document(doc)

        assert len(result) == 2


class TestRAGPipelineQuery:
    """RAGPipeline.query() のテスト。"""

    @pytest.fixture
    def index_manager(self) -> AsyncMock:
        return AsyncMock(spec=IndexManager)

    @pytest.fixture
    def retriever(self) -> AsyncMock:
        mock = AsyncMock(spec=Retriever)
        mock.search.return_value = [make_retrieved_chunk()]
        return mock

    @pytest.fixture
    def mock_generator(self) -> AsyncMock:
        mock = AsyncMock(spec=Generator)
        mock.generate.return_value = GenerationResult(
            answer="回答",
            sources=[make_retrieved_chunk()],
            model="test-model",
        )
        return mock

    @pytest.fixture
    def pipeline(
        self,
        index_manager: AsyncMock,
        retriever: AsyncMock,
        mock_generator: AsyncMock,
    ) -> RAGPipeline:
        return RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=mock_generator,
        )

    async def test_returns_generation_result(self, pipeline: RAGPipeline) -> None:
        """GenerationResult が返ること。"""
        result = await pipeline.query("質問")

        assert isinstance(result, GenerationResult)

    async def test_calls_retriever_with_query_and_top_k(
        self, pipeline: RAGPipeline, retriever: AsyncMock
    ) -> None:
        """Retriever.search がクエリと top_k で呼ばれること。"""
        await pipeline.query("テスト質問", top_k=3)

        retriever.search.assert_awaited_once_with("テスト質問", top_k=3, document_id=None)

    async def test_calls_generator_with_query_and_chunks(
        self, pipeline: RAGPipeline, retriever: AsyncMock, mock_generator: AsyncMock
    ) -> None:
        """Generator.generate がクエリとチャンクで呼ばれること。"""
        retrieved = [make_retrieved_chunk(content="コンテキスト")]
        retriever.search.return_value = retrieved

        await pipeline.query("質問")

        mock_generator.generate.assert_awaited_once_with("質問", retrieved)

    async def test_wraps_retrieval_error(self, pipeline: RAGPipeline, retriever: AsyncMock) -> None:
        """RetrievalError が RAGPipelineError でラップされること。"""
        retrieval_error = RetrievalError("search failed")
        retriever.search.side_effect = retrieval_error

        with pytest.raises(RAGPipelineError) as exc_info:
            await pipeline.query("質問")

        assert exc_info.value.__cause__ is retrieval_error

    async def test_wraps_generation_error(
        self, pipeline: RAGPipeline, mock_generator: AsyncMock
    ) -> None:
        """GenerationError が RAGPipelineError でラップされること。"""
        gen_error = GenerationError("generate failed")
        mock_generator.generate.side_effect = gen_error

        with pytest.raises(RAGPipelineError) as exc_info:
            await pipeline.query("質問")

        assert exc_info.value.__cause__ is gen_error

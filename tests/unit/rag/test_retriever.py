"""Tests for RetrievedChunk model and Retriever.search()."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from src.db.vector_store import VectorStoreError
from src.rag.embedder import EmbeddingError
from src.rag.retriever import RetrievedChunk, Retriever

from .conftest import EMBEDDING_DIM, make_chunk


class TestRetrievedChunk:
    """RetrievedChunk モデルのテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """必須フィールドで生成できること。"""
        doc_id = uuid.uuid4()
        chunk = RetrievedChunk(
            content="テスト内容",
            chunk_index=0,
            document_id=doc_id,
        )

        assert chunk.content == "テスト内容"
        assert chunk.chunk_index == 0
        assert chunk.document_id == doc_id


class TestRetrieverSearch:
    """Retriever.search() のテスト。"""

    @pytest.fixture
    def embedder(self) -> AsyncMock:
        mock = AsyncMock()
        mock.embed.return_value = [0.1] * EMBEDDING_DIM
        return mock

    @pytest.fixture
    def vector_store(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def retriever(self, embedder: AsyncMock, vector_store: AsyncMock) -> Retriever:
        return Retriever(embedder=embedder, vector_store=vector_store)

    async def test_returns_retrieved_chunks(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """Chunk が RetrievedChunk に変換されて返ること。"""
        doc_id = uuid.uuid4()
        chunk = make_chunk(document_id=doc_id, chunk_index=0, content="結果テキスト")
        vector_store.search.return_value = [chunk]

        result = await retriever.search("クエリ")

        assert len(result) == 1
        assert isinstance(result[0], RetrievedChunk)
        assert result[0].content == "結果テキスト"
        assert result[0].chunk_index == 0
        assert result[0].document_id == doc_id

    async def test_calls_embedder_with_query(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """Embedder.embed がクエリ文字列で呼ばれること。"""
        vector_store.search.return_value = []

        await retriever.search("テストクエリ")

        embedder.embed.assert_awaited_once_with("テストクエリ")

    async def test_calls_vector_store_with_embedding_and_top_k(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """VectorStore.search が embedding と top_k で呼ばれること。"""
        vector_store.search.return_value = []

        await retriever.search("クエリ", top_k=3)

        vector_store.search.assert_awaited_once_with(
            query_embedding=[0.1] * EMBEDDING_DIM,
            top_k=3,
            document_id=None,
        )

    async def test_passes_document_id_to_vector_store(
        self, retriever: Retriever, embedder: AsyncMock, vector_store: AsyncMock
    ) -> None:
        """document_id が VectorStore.search に渡されること。"""
        doc_id = uuid.uuid4()
        vector_store.search.return_value = []

        await retriever.search("クエリ", document_id=doc_id)

        vector_store.search.assert_awaited_once_with(
            query_embedding=[0.1] * EMBEDDING_DIM,
            top_k=5,
            document_id=doc_id,
        )

    async def test_returns_empty_list_when_no_results(
        self, retriever: Retriever, vector_store: AsyncMock
    ) -> None:
        """検索結果がない場合に空リストが返ること。"""
        vector_store.search.return_value = []

        result = await retriever.search("クエリ")

        assert result == []

    async def test_wraps_embedding_error(self, retriever: Retriever, embedder: AsyncMock) -> None:
        """EmbeddingError が RetrievalError でラップされること。"""
        from src.rag.retriever import RetrievalError

        embedding_error = EmbeddingError("connection failed")
        embedder.embed.side_effect = embedding_error

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.search("クエリ")

        assert exc_info.value.__cause__ is embedding_error

    async def test_wraps_vector_store_error(
        self, retriever: Retriever, vector_store: AsyncMock
    ) -> None:
        """VectorStoreError が RetrievalError でラップされること。"""
        from src.rag.retriever import RetrievalError

        vs_error = VectorStoreError("query failed")
        vector_store.search.side_effect = vs_error

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.search("クエリ")

        assert exc_info.value.__cause__ is vs_error

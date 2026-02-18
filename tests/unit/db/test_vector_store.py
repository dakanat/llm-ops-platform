"""Tests for pgvector-based vector store operations."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.models import Chunk
from src.db.vector_store import VectorStore, VectorStoreError

EMBEDDING_DIM = 1024


def _make_chunk(
    document_id: uuid.UUID | None = None,
    chunk_index: int = 0,
    content: str = "テストチャンク",
    with_embedding: bool = True,
) -> Chunk:
    """テスト用の Chunk インスタンスを生成。"""
    return Chunk(
        id=uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        content=content,
        chunk_index=chunk_index,
        embedding=[0.1] * EMBEDDING_DIM if with_embedding else None,
    )


def _make_query_embedding(dimensions: int = EMBEDDING_DIM) -> list[float]:
    """テスト用のクエリ embedding を生成。"""
    return [0.1] * dimensions


class TestVectorStoreInit:
    """VectorStore の初期化テスト。"""

    def test_creates_with_async_session(self) -> None:
        """AsyncSession で生成できること。"""
        session = AsyncMock(spec=AsyncSession)
        store = VectorStore(session)

        assert store._session is session


class TestVectorStoreSaveChunks:
    """VectorStore.save_chunks() のテスト。"""

    @pytest.fixture
    def session(self) -> AsyncMock:
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def store(self, session: AsyncMock) -> VectorStore:
        return VectorStore(session)

    async def test_save_chunks_calls_add_all(self, store: VectorStore, session: AsyncMock) -> None:
        """add_all がチャンクリストで呼ばれること。"""
        chunks = [_make_chunk(), _make_chunk(chunk_index=1)]

        await store.save_chunks(chunks)

        session.add_all.assert_called_once_with(chunks)

    async def test_save_chunks_calls_flush(self, store: VectorStore, session: AsyncMock) -> None:
        """flush が呼ばれること。"""
        chunks = [_make_chunk()]

        await store.save_chunks(chunks)

        session.flush.assert_awaited_once()

    async def test_save_chunks_raises_on_empty_list(self, store: VectorStore) -> None:
        """空リストで VectorStoreError が発生すること。"""
        with pytest.raises(VectorStoreError, match="empty"):
            await store.save_chunks([])

    async def test_save_chunks_raises_on_db_error(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """DB エラーで VectorStoreError が発生し、原因チェインが保持されること。"""
        db_error = SQLAlchemyError("connection failed")
        session.flush.side_effect = db_error

        with pytest.raises(VectorStoreError) as exc_info:
            await store.save_chunks([_make_chunk()])

        assert exc_info.value.__cause__ is db_error


class TestVectorStoreSearch:
    """VectorStore.search() のテスト。"""

    @pytest.fixture
    def session(self) -> AsyncMock:
        mock_session = AsyncMock(spec=AsyncSession)
        return mock_session

    @pytest.fixture
    def store(self, session: AsyncMock) -> VectorStore:
        return VectorStore(session)

    def _setup_exec_result(self, session: AsyncMock, chunks: list[Chunk]) -> None:
        """session.exec の戻り値を設定。"""
        mock_result = MagicMock()
        mock_result.all.return_value = chunks
        session.exec.return_value = mock_result

    async def test_search_returns_list_of_chunks(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """Chunk のリストが返ること。"""
        expected_chunks = [_make_chunk(), _make_chunk(chunk_index=1)]
        self._setup_exec_result(session, expected_chunks)

        result = await store.search(_make_query_embedding())

        assert result == expected_chunks
        assert all(isinstance(c, Chunk) for c in result)

    async def test_search_default_top_k_is_5(self, store: VectorStore, session: AsyncMock) -> None:
        """デフォルト top_k=5 で呼び出せること。"""
        self._setup_exec_result(session, [])

        result = await store.search(_make_query_embedding())

        assert isinstance(result, list)
        session.exec.assert_awaited_once()

    async def test_search_respects_custom_top_k(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """カスタム top_k が反映されること。"""
        self._setup_exec_result(session, [_make_chunk()])

        result = await store.search(_make_query_embedding(), top_k=3)

        assert isinstance(result, list)
        session.exec.assert_awaited_once()

    async def test_search_filters_by_document_id(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """document_id フィルタが適用されること。"""
        doc_id = uuid.uuid4()
        chunk = _make_chunk(document_id=doc_id)
        self._setup_exec_result(session, [chunk])

        result = await store.search(_make_query_embedding(), document_id=doc_id)

        assert result == [chunk]
        session.exec.assert_awaited_once()

    async def test_search_without_document_id_searches_all(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """document_id=None で全検索されること。"""
        chunks = [_make_chunk(), _make_chunk(chunk_index=1)]
        self._setup_exec_result(session, chunks)

        result = await store.search(_make_query_embedding(), document_id=None)

        assert len(result) == 2
        session.exec.assert_awaited_once()

    async def test_search_raises_on_invalid_embedding_dimensions(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """次元数不正で VectorStoreError が発生すること。"""
        wrong_dim_embedding = [0.1] * 512

        with pytest.raises(VectorStoreError, match="1024"):
            await store.search(wrong_dim_embedding)

        session.exec.assert_not_awaited()

    async def test_search_raises_on_zero_top_k(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """top_k=0 で VectorStoreError が発生すること。"""
        with pytest.raises(VectorStoreError, match="top_k"):
            await store.search(_make_query_embedding(), top_k=0)

        session.exec.assert_not_awaited()

    async def test_search_raises_on_negative_top_k(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """負の top_k で VectorStoreError が発生すること。"""
        with pytest.raises(VectorStoreError, match="top_k"):
            await store.search(_make_query_embedding(), top_k=-1)

        session.exec.assert_not_awaited()

    async def test_search_raises_on_empty_embedding(
        self, store: VectorStore, session: AsyncMock
    ) -> None:
        """空 embedding で VectorStoreError が発生すること。"""
        with pytest.raises(VectorStoreError, match="empty"):
            await store.search([])

        session.exec.assert_not_awaited()

    async def test_search_raises_on_db_error(self, store: VectorStore, session: AsyncMock) -> None:
        """DB エラーで VectorStoreError が発生し、原因チェインが保持されること。"""
        db_error = SQLAlchemyError("query failed")
        session.exec.side_effect = db_error

        with pytest.raises(VectorStoreError) as exc_info:
            await store.search(_make_query_embedding())

        assert exc_info.value.__cause__ is db_error

"""Tests for IndexManager.index_document()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.db.models import Chunk
from src.db.vector_store import VectorStoreError
from src.rag.chunker import RecursiveCharacterSplitter
from src.rag.embedder import EmbeddingError
from src.rag.index_manager import IndexingError, IndexManager
from src.rag.preprocessor import Preprocessor

from .conftest import EMBEDDING_DIM, make_document, make_text_chunk


class TestIndexManagerIndexDocument:
    """IndexManager.index_document() のテスト。"""

    @pytest.fixture
    def preprocessor(self) -> Preprocessor:
        return Preprocessor()

    @pytest.fixture
    def chunker(self) -> MagicMock:
        mock = MagicMock(spec=RecursiveCharacterSplitter)
        return mock

    @pytest.fixture
    def embedder(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def vector_store(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def index_manager(
        self,
        preprocessor: Preprocessor,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> IndexManager:
        return IndexManager(
            preprocessor=preprocessor,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

    async def test_returns_list_of_chunks(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> None:
        """Chunk のリストが返ること。"""
        doc = make_document(content="テスト文書")
        chunker.split.return_value = [make_text_chunk(content="テスト文書")]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        result = await index_manager.index_document(doc)

        assert len(result) == 1
        assert isinstance(result[0], Chunk)

    async def test_preprocesses_document_content(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """前処理が適用されること (全角→半角)。"""
        doc = make_document(content="Ｈｅｌｌｏ")
        chunker.split.return_value = [make_text_chunk(content="Hello")]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        await index_manager.index_document(doc)

        chunker.split.assert_called_once_with("Hello")

    async def test_chunks_preprocessed_text(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """前処理済みテキストがチャンカーに渡されること。"""
        doc = make_document(content="テスト文書")
        chunker.split.return_value = [make_text_chunk(content="テスト文書")]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        await index_manager.index_document(doc)

        chunker.split.assert_called_once_with("テスト文書")

    async def test_embeds_chunk_contents(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """チャンク内容が embed_batch に渡されること。"""
        doc = make_document()
        chunker.split.return_value = [
            make_text_chunk(content="チャンク1"),
            make_text_chunk(content="チャンク2", index=1, start=5),
        ]
        embedder.embed_batch.return_value = [
            [0.1] * EMBEDDING_DIM,
            [0.2] * EMBEDDING_DIM,
        ]

        await index_manager.index_document(doc)

        embedder.embed_batch.assert_awaited_once_with(["チャンク1", "チャンク2"])

    async def test_saves_chunks_to_vector_store(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> None:
        """VectorStore.save_chunks が呼ばれること。"""
        doc = make_document()
        chunker.split.return_value = [make_text_chunk()]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]

        await index_manager.index_document(doc)

        vector_store.save_chunks.assert_awaited_once()
        saved_chunks = vector_store.save_chunks.call_args[0][0]
        assert len(saved_chunks) == 1
        assert saved_chunks[0].document_id == doc.id

    async def test_wraps_embedding_error(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
    ) -> None:
        """EmbeddingError が IndexingError でラップされること。"""
        doc = make_document()
        chunker.split.return_value = [make_text_chunk()]
        embedding_error = EmbeddingError("embed failed")
        embedder.embed_batch.side_effect = embedding_error

        with pytest.raises(IndexingError) as exc_info:
            await index_manager.index_document(doc)

        assert exc_info.value.__cause__ is embedding_error

    async def test_wraps_vector_store_error(
        self,
        index_manager: IndexManager,
        chunker: MagicMock,
        embedder: AsyncMock,
        vector_store: AsyncMock,
    ) -> None:
        """VectorStoreError が IndexingError でラップされること。"""
        doc = make_document()
        chunker.split.return_value = [make_text_chunk()]
        embedder.embed_batch.return_value = [[0.1] * EMBEDDING_DIM]
        vs_error = VectorStoreError("save failed")
        vector_store.save_chunks.side_effect = vs_error

        with pytest.raises(IndexingError) as exc_info:
            await index_manager.index_document(doc)

        assert exc_info.value.__cause__ is vs_error

    async def test_empty_content_raises_error(
        self,
        index_manager: IndexManager,
    ) -> None:
        """空コンテンツの Document で IndexingError が発生すること。"""
        doc = make_document(content="")

        with pytest.raises(IndexingError):
            await index_manager.index_document(doc)

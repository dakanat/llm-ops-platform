"""Vector similarity search retriever for RAG pipeline."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from src.db.models import Chunk
from src.db.vector_store import VectorStore, VectorStoreError
from src.rag.embedder import Embedder, EmbeddingError


class RetrievalError(Exception):
    """検索処理に関するエラー。"""


class RetrievedChunk(BaseModel):
    """検索結果のチャンク。

    Attributes:
        content: チャンクのテキスト内容。
        chunk_index: ドキュメント内のチャンク通し番号。
        document_id: 元ドキュメントの ID。
    """

    content: str
    chunk_index: int
    document_id: uuid.UUID


class Retriever:
    """ベクトル類似検索ラッパー。

    クエリテキストを Embedding に変換し、VectorStore で類似チャンクを検索する。
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: uuid.UUID | None = None,
    ) -> list[RetrievedChunk]:
        """クエリに類似するチャンクを検索する。

        Args:
            query: 検索クエリテキスト。
            top_k: 返却する最大件数。
            document_id: 特定ドキュメントに限定する場合の ID。

        Returns:
            類似度順の RetrievedChunk リスト。

        Raises:
            RetrievalError: Embedding 生成または検索に失敗した場合。
        """
        try:
            query_embedding = await self._embedder.embed(query)
        except EmbeddingError as e:
            raise RetrievalError(str(e)) from e

        try:
            chunks: list[Chunk] = await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id=document_id,
            )
        except VectorStoreError as e:
            raise RetrievalError(str(e)) from e

        return [
            RetrievedChunk(
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
            )
            for chunk in chunks
        ]

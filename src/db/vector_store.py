"""pgvector-based vector store for embedding storage and similarity search."""

from __future__ import annotations

import uuid

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.models import Chunk

EMBEDDING_DIMENSIONS = 1024


class VectorStoreError(Exception):
    """Error raised by VectorStore operations."""


class VectorStore:
    """Manages vector embedding storage and cosine similarity search using pgvector."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        """Save chunks to the database.

        Uses flush() instead of commit() so the caller can control the transaction.

        Args:
            chunks: List of Chunk instances to persist.

        Raises:
            VectorStoreError: If chunks list is empty or a database error occurs.
        """
        if not chunks:
            msg = "chunks list must not be empty"
            raise VectorStoreError(msg)

        try:
            self._session.add_all(chunks)
            await self._session.flush()
        except Exception as e:
            raise VectorStoreError(str(e)) from e

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: uuid.UUID | None = None,
    ) -> list[Chunk]:
        """Search for similar chunks using cosine distance.

        Args:
            query_embedding: Query vector (must be 1024 dimensions).
            top_k: Number of results to return. Must be positive.
            document_id: Optional filter to restrict search to a specific document.

        Returns:
            List of Chunk instances ordered by cosine similarity (most similar first).

        Raises:
            VectorStoreError: If parameters are invalid or a database error occurs.
        """
        if not query_embedding:
            msg = "query_embedding must not be empty"
            raise VectorStoreError(msg)

        if len(query_embedding) != EMBEDDING_DIMENSIONS:
            msg = (
                f"query_embedding must have {EMBEDDING_DIMENSIONS} dimensions, "
                f"got {len(query_embedding)}"
            )
            raise VectorStoreError(msg)

        if top_k <= 0:
            msg = f"top_k must be positive, got {top_k}"
            raise VectorStoreError(msg)

        try:
            stmt = (
                select(Chunk)
                .where(Chunk.embedding.is_not(None))  # type: ignore[union-attr]
                .order_by(Chunk.embedding.cosine_distance(query_embedding))  # type: ignore[union-attr]
                .limit(top_k)
            )

            if document_id is not None:
                stmt = stmt.where(Chunk.document_id == document_id)

            result = await self._session.exec(stmt)
            return list(result.all())
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(str(e)) from e

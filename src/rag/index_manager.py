"""Document indexing manager for RAG pipeline."""

from __future__ import annotations

from src.db.models import Chunk, Document
from src.db.vector_store import VectorStore, VectorStoreError
from src.rag.chunker import RecursiveCharacterSplitter
from src.rag.embedder import Embedder, EmbeddingError
from src.rag.preprocessor import PreprocessingError, Preprocessor


class IndexingError(Exception):
    """インデックス処理に関するエラー。"""


class IndexManager:
    """ドキュメントのインデックス管理。

    前処理 → チャンク分割 → Embedding 生成 → ベクトルストア保存の
    一連のフローを統合する。
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        chunker: RecursiveCharacterSplitter,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._preprocessor = preprocessor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    async def index_document(self, document: Document) -> list[Chunk]:
        """ドキュメントをインデックスに登録する。

        処理フロー:
        1. テキスト前処理 (NFKC 正規化、空白正規化)
        2. チャンク分割
        3. Embedding 生成 (バッチ)
        4. Chunk モデル構築
        5. ベクトルストアに保存

        Args:
            document: インデックスに登録するドキュメント。

        Returns:
            保存された Chunk のリスト。

        Raises:
            IndexingError: 前処理、Embedding 生成、保存のいずれかに失敗した場合。
        """
        # 1. 前処理
        try:
            normalized_text = self._preprocessor.normalize(document.content)
        except PreprocessingError as e:
            raise IndexingError(str(e)) from e

        # 2. チャンク分割
        text_chunks = self._chunker.split(normalized_text)

        # 3. Embedding 生成
        contents = [tc.content for tc in text_chunks]
        try:
            embeddings = await self._embedder.embed_batch(contents)
        except EmbeddingError as e:
            raise IndexingError(str(e)) from e

        # 4. Chunk モデル構築
        chunks = [
            Chunk(
                document_id=document.id,
                content=tc.content,
                chunk_index=tc.index,
                embedding=emb,
            )
            for tc, emb in zip(text_chunks, embeddings, strict=True)
        ]

        # 5. ベクトルストアに保存
        try:
            await self._vector_store.save_chunks(chunks)
        except VectorStoreError as e:
            raise IndexingError(str(e)) from e

        return chunks

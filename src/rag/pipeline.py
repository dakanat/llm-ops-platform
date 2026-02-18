"""RAG pipeline orchestrator."""

from __future__ import annotations

import uuid

from src.db.models import Chunk, Document
from src.rag.generator import GenerationError, GenerationResult, Generator
from src.rag.index_manager import IndexingError, IndexManager
from src.rag.retriever import RetrievalError, Retriever


class RAGPipelineError(Exception):
    """RAG パイプライン処理に関するエラー。"""


class RAGPipeline:
    """RAG パイプラインオーケストレーター。

    IndexManager, Retriever, Generator を統合し、
    ドキュメント登録とクエリ応答の一連のフローを提供する。
    """

    def __init__(
        self,
        index_manager: IndexManager,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self._index_manager = index_manager
        self._retriever = retriever
        self._generator = generator

    async def index_document(self, document: Document) -> list[Chunk]:
        """ドキュメントをインデックスに登録する。

        Args:
            document: 登録するドキュメント。

        Returns:
            保存された Chunk のリスト。

        Raises:
            RAGPipelineError: インデックス処理に失敗した場合。
        """
        try:
            return await self._index_manager.index_document(document)
        except IndexingError as e:
            raise RAGPipelineError(str(e)) from e

    async def query(
        self,
        query: str,
        top_k: int = 5,
        document_id: uuid.UUID | None = None,
    ) -> GenerationResult:
        """クエリに対して検索 → 回答生成を行う。

        Args:
            query: ユーザーの質問テキスト。
            top_k: 検索で取得するチャンク数。
            document_id: 特定ドキュメントに限定する場合の ID。

        Returns:
            回答テキスト、ソース、モデル名を含む GenerationResult。

        Raises:
            RAGPipelineError: 検索または回答生成に失敗した場合。
        """
        try:
            chunks = await self._retriever.search(query, top_k=top_k, document_id=document_id)
        except RetrievalError as e:
            raise RAGPipelineError(str(e)) from e

        try:
            return await self._generator.generate(query, chunks)
        except GenerationError as e:
            raise RAGPipelineError(str(e)) from e

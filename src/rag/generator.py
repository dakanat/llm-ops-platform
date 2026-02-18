"""LLM answer generator with source citation for RAG pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.llm.providers.base import (
    ChatMessage,
    LLMProvider,
    Role,
    TokenUsage,
)
from src.rag.retriever import RetrievedChunk


class GenerationError(Exception):
    """回答生成に関するエラー。"""


class GenerationResult(BaseModel):
    """RAG 回答生成の結果。

    Attributes:
        answer: LLM が生成した回答テキスト。
        sources: 回答の根拠となったチャンク群。
        model: 使用した LLM モデル名。
        usage: トークン使用量 (取得できた場合)。
    """

    answer: str
    sources: list[RetrievedChunk]
    model: str
    usage: TokenUsage | None = None


class Generator:
    """コンテキストチャンクとクエリから LLM で回答を生成する。

    チャンクを番号付きコンテキストとしてフォーマットし、
    LLM に質問とともに送信して回答を得る。
    """

    DEFAULT_SYSTEM_PROMPT: str = (
        "あなたは質問応答アシスタントです。"
        "提供されたコンテキストに基づいて質問に回答してください。"
        "回答にはコンテキストの番号を [1], [2] のように引用してください。"
        "コンテキストに情報がない場合は、その旨を正直に伝えてください。"
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        system_prompt: str | None = None,
    ) -> None:
        self._provider = llm_provider
        self._model = model
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    async def generate(
        self, query: str, chunks: list[RetrievedChunk], **kwargs: Any
    ) -> GenerationResult:
        """クエリとチャンクから回答を生成する。

        Args:
            query: ユーザーの質問テキスト。
            chunks: 検索で取得したチャンク群。
            **kwargs: LLM プロバイダに渡す追加パラメータ。

        Returns:
            回答テキスト、ソース、モデル名を含む GenerationResult。

        Raises:
            GenerationError: LLM 呼び出しに失敗した場合。
        """
        context = self._build_context(chunks)
        messages = self._build_messages(query, context)

        try:
            response = await self._provider.complete(
                messages=messages,
                model=self._model,
                **kwargs,
            )
        except Exception as e:
            raise GenerationError(str(e)) from e

        return GenerationResult(
            answer=response.content,
            sources=chunks,
            model=response.model,
            usage=response.usage,
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """チャンクを番号付きコンテキスト文字列に変換する。

        Args:
            chunks: コンテキストとして使用するチャンク群。

        Returns:
            ``[1] チャンク内容\\n\\n[2] チャンク内容...`` 形式の文字列。
            空リストの場合は空文字列。
        """
        if not chunks:
            return ""
        return "\n\n".join(f"[{i + 1}] {chunk.content}" for i, chunk in enumerate(chunks))

    def _build_messages(self, query: str, context: str) -> list[ChatMessage]:
        """LLM に送信するメッセージリストを構築する。

        Args:
            query: ユーザーの質問テキスト。
            context: 番号付きコンテキスト文字列。

        Returns:
            system メッセージと user メッセージのリスト。
        """
        return [
            ChatMessage(role=Role.system, content=self._system_prompt),
            ChatMessage(
                role=Role.user,
                content=f"コンテキスト:\n{context}\n\n質問: {query}",
            ),
        ]

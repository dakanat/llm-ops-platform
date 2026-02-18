"""RAG 検索ツール。

RAGPipeline を利用してベクトル検索と回答生成を実行する。
"""

from __future__ import annotations

from typing import ClassVar

from src.agent.tools.base import ToolResult
from src.rag.pipeline import RAGPipeline, RAGPipelineError


class SearchTool:
    """RAG パイプラインによるドキュメント検索ツール。

    クエリテキストを受け取り、RAGPipeline で類似チャンクを検索して
    回答を生成する。結果にはソースチャンクの引用を含む。
    """

    name: ClassVar[str] = "search"
    description: ClassVar[str] = (
        "Search documents using RAG pipeline. "
        "Input is a natural language query. "
        "Returns an answer with source references."
    )

    def __init__(self, pipeline: RAGPipeline, top_k: int = 5) -> None:
        """SearchTool を初期化する。

        Args:
            pipeline: 検索と回答生成に使用する RAGPipeline インスタンス。
            top_k: 検索で取得するチャンク数。デフォルトは 5。
        """
        self._pipeline = pipeline
        self._top_k = top_k

    async def execute(self, input_text: str) -> ToolResult:
        """クエリを実行して検索結果と回答を返す。

        エラー時は例外を raise せず ToolResult.error に格納する。

        Args:
            input_text: 検索クエリテキスト。

        Returns:
            回答とソース情報を含む ToolResult。
        """
        if not input_text.strip():
            return ToolResult(output="", error="Empty query")

        try:
            result = await self._pipeline.query(input_text, top_k=self._top_k)
        except RAGPipelineError as e:
            return ToolResult(output="", error=str(e))

        return ToolResult(output=self._format_output(result))

    @staticmethod
    def _format_output(result: object) -> str:
        """GenerationResult を可読文字列に変換する。

        Args:
            result: RAGPipeline.query() の戻り値。

        Returns:
            回答とソース引用を含むフォーマット済み文字列。
        """
        from src.rag.generator import GenerationResult

        assert isinstance(result, GenerationResult)

        lines: list[str] = [result.answer]

        if result.sources:
            lines.append("")
            lines.append("Sources:")
            for i, src in enumerate(result.sources, 1):
                lines.append(f"  [{i}] {src.content}")

        return "\n".join(lines)

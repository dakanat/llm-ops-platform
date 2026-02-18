"""フォールバック戦略。

ツール実行失敗時のリトライと、全試行失敗時の degradation を管理する。
"""

from __future__ import annotations

from pydantic import BaseModel

from src.agent.tools.base import Tool, ToolResult


class FallbackResult(BaseModel):
    """フォールバック実行結果。

    Attributes:
        tool_result: ツール実行結果 (成功 or 最後のエラー)。
        retries_attempted: 実行されたリトライ回数 (初回実行は含まない)。
        degraded: 全試行失敗により直接回答への切り替えが必要かどうか。
    """

    tool_result: ToolResult
    retries_attempted: int
    degraded: bool


class FallbackStrategy:
    """ツール実行のフォールバック戦略。

    初回実行 + 最大 max_retries 回のリトライを行い、
    全試行失敗時は degraded=True を返して Runtime に
    直接回答への切り替えを通知する。
    """

    def __init__(self, max_retries: int = 2) -> None:
        self._max_retries = max_retries

    async def execute_with_fallback(self, tool: Tool, input_text: str) -> FallbackResult:
        """ツールをリトライ付きで実行する。

        ToolResult.is_error が True、または例外発生時にリトライする。
        例外は ToolResult(output="", error=str(e)) に変換する。

        Args:
            tool: 実行するツール。
            input_text: ツールへの入力テキスト。

        Returns:
            FallbackResult。
        """
        last_result: ToolResult | None = None
        total_attempts = 1 + self._max_retries

        for attempt in range(total_attempts):
            try:
                result = await tool.execute(input_text)
            except Exception as e:
                result = ToolResult(output="", error=str(e))

            if not result.is_error:
                return FallbackResult(
                    tool_result=result,
                    retries_attempted=attempt,
                    degraded=False,
                )

            last_result = result

        # 全試行失敗
        assert last_result is not None
        return FallbackResult(
            tool_result=last_result,
            retries_attempted=self._max_retries,
            degraded=True,
        )

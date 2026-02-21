"""Tool Protocol と ToolResult モデル。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class ToolResult(BaseModel):
    """ツール実行結果。

    Attributes:
        output: 実行結果の文字列表現。
        error: エラーメッセージ。成功時は None。
        metadata: ツール固有の構造化データ。テンプレートへの受け渡し用。
    """

    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        """エラーが存在するかどうかを返す。"""
        return self.error is not None


@runtime_checkable
class Tool(Protocol):
    """すべてのツールが満たすべき Protocol。"""

    name: str
    description: str

    async def execute(self, input_text: str) -> ToolResult: ...

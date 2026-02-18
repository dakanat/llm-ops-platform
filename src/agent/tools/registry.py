"""ツールレジストリ。

ツールの登録、取得、一覧を管理する。
"""

from __future__ import annotations

from src.agent import DuplicateToolError, ToolNotFoundError
from src.agent.tools.base import Tool


class ToolRegistry:
    """ツールのレジストリ。

    ツールを名前で管理し、重複登録を拒否する。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """ツールを登録する。

        Args:
            tool: 登録するツール。

        Raises:
            DuplicateToolError: 同名ツールが既に登録されている場合。
        """
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """名前でツールを取得する。

        Args:
            name: ツール名。

        Returns:
            対応する Tool インスタンス。

        Raises:
            ToolNotFoundError: ツールが見つからない場合。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool not found: {name}")
        return tool

    def has(self, name: str) -> bool:
        """ツールが登録済みかどうかを返す。

        Args:
            name: ツール名。

        Returns:
            登録済みなら True。
        """
        return name in self._tools

    def list_tools(self) -> list[str]:
        """登録済みツール名のソート済みリストを返す。

        Returns:
            ソート済みのツール名リスト。
        """
        return sorted(self._tools.keys())

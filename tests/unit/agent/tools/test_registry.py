"""ToolRegistry のユニットテスト。

ツール登録、取得、重複エラー、一覧、存在チェックを検証する。
"""

from __future__ import annotations

import pytest
from src.agent import DuplicateToolError, ToolNotFoundError
from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# テスト用スタブツール
# ---------------------------------------------------------------------------
class _StubTool:
    """テスト用の最小 Tool 実装。"""

    def __init__(self, name: str = "stub", description: str = "A stub tool") -> None:
        self.name = name
        self.description = description

    async def execute(self, input_text: str) -> ToolResult:
        return ToolResult(output=f"stub: {input_text}")


class _AnotherStubTool:
    """2つ目のスタブツール。"""

    def __init__(self) -> None:
        self.name = "another"
        self.description = "Another stub tool"

    async def execute(self, input_text: str) -> ToolResult:
        return ToolResult(output=f"another: {input_text}")


# ---------------------------------------------------------------------------
# TestToolRegistryInit
# ---------------------------------------------------------------------------
class TestToolRegistryInit:
    """ToolRegistry の初期化を検証する。"""

    def test_instantiation(self) -> None:
        registry = ToolRegistry()
        assert isinstance(registry, ToolRegistry)

    def test_initial_state_is_empty(self) -> None:
        registry = ToolRegistry()
        assert registry.list_tools() == []


# ---------------------------------------------------------------------------
# TestToolRegistryRegister
# ---------------------------------------------------------------------------
class TestToolRegistryRegister:
    """ツール登録を検証する。"""

    def test_register_single_tool(self) -> None:
        registry = ToolRegistry()
        tool = _StubTool()
        registry.register(tool)
        assert registry.has("stub")

    def test_register_duplicate_raises_error(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool())
        with pytest.raises(DuplicateToolError, match="stub"):
            registry.register(_StubTool())

    def test_register_multiple_different_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool())
        registry.register(_AnotherStubTool())
        assert registry.has("stub")
        assert registry.has("another")


# ---------------------------------------------------------------------------
# TestToolRegistryGet
# ---------------------------------------------------------------------------
class TestToolRegistryGet:
    """ツール取得を検証する。"""

    def test_get_registered_tool(self) -> None:
        registry = ToolRegistry()
        tool = _StubTool()
        registry.register(tool)
        result = registry.get("stub")
        assert isinstance(result, Tool)

    def test_get_returns_same_instance(self) -> None:
        registry = ToolRegistry()
        tool = _StubTool()
        registry.register(tool)
        assert registry.get("stub") is tool

    def test_get_unregistered_raises_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            registry.get("nonexistent")


# ---------------------------------------------------------------------------
# TestToolRegistryHas
# ---------------------------------------------------------------------------
class TestToolRegistryHas:
    """ツール存在チェックを検証する。"""

    def test_has_returns_true_for_registered(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool())
        assert registry.has("stub") is True

    def test_has_returns_false_for_unregistered(self) -> None:
        registry = ToolRegistry()
        assert registry.has("nonexistent") is False


# ---------------------------------------------------------------------------
# TestToolRegistryListTools
# ---------------------------------------------------------------------------
class TestToolRegistryListTools:
    """ツール一覧の取得を検証する。"""

    def test_list_tools_empty(self) -> None:
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_list_tools_returns_sorted_names(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool(name="charlie", description="c"))
        registry.register(_StubTool(name="alpha", description="a"))
        registry.register(_StubTool(name="bravo", description="b"))
        assert registry.list_tools() == ["alpha", "bravo", "charlie"]

    def test_list_tools_returns_new_list_each_time(self) -> None:
        registry = ToolRegistry()
        registry.register(_StubTool())
        result1 = registry.list_tools()
        result2 = registry.list_tools()
        assert result1 == result2
        assert result1 is not result2

"""Tool Protocol、ToolResult モデル、例外階層のユニットテスト。"""

from __future__ import annotations

import pytest
from src.agent import (
    DuplicateToolError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from src.agent.tools.base import Tool, ToolResult


# ---------------------------------------------------------------------------
# TestToolErrorHierarchy
# ---------------------------------------------------------------------------
class TestToolErrorHierarchy:
    """例外クラスの継承関係を検証する。"""

    def test_tool_error_is_exception(self) -> None:
        assert issubclass(ToolError, Exception)

    def test_tool_not_found_error_is_tool_error(self) -> None:
        assert issubclass(ToolNotFoundError, ToolError)

    def test_duplicate_tool_error_is_tool_error(self) -> None:
        assert issubclass(DuplicateToolError, ToolError)

    def test_tool_execution_error_is_tool_error(self) -> None:
        assert issubclass(ToolExecutionError, ToolError)

    def test_tool_error_can_be_raised_with_message(self) -> None:
        with pytest.raises(ToolError, match="test message"):
            raise ToolError("test message")

    def test_tool_not_found_error_can_be_raised(self) -> None:
        with pytest.raises(ToolNotFoundError, match="unknown_tool"):
            raise ToolNotFoundError("unknown_tool")

    def test_duplicate_tool_error_can_be_raised(self) -> None:
        with pytest.raises(DuplicateToolError, match="calculator"):
            raise DuplicateToolError("calculator")

    def test_tool_execution_error_can_be_raised(self) -> None:
        with pytest.raises(ToolExecutionError, match="division by zero"):
            raise ToolExecutionError("division by zero")


# ---------------------------------------------------------------------------
# TestToolResult
# ---------------------------------------------------------------------------
class TestToolResult:
    """ToolResult モデルのフィールドと is_error プロパティを検証する。"""

    def test_success_result(self) -> None:
        result = ToolResult(output="42")
        assert result.output == "42"
        assert result.error is None

    def test_error_result(self) -> None:
        result = ToolResult(output="", error="division by zero")
        assert result.output == ""
        assert result.error == "division by zero"

    def test_is_error_false_for_success(self) -> None:
        result = ToolResult(output="ok")
        assert result.is_error is False

    def test_is_error_true_when_error_set(self) -> None:
        result = ToolResult(output="", error="fail")
        assert result.is_error is True

    def test_is_error_false_when_error_is_none(self) -> None:
        result = ToolResult(output="data", error=None)
        assert result.is_error is False


# ---------------------------------------------------------------------------
# TestToolProtocol
# ---------------------------------------------------------------------------
class TestToolProtocol:
    """Tool Protocol の isinstance チェックを検証する。"""

    def test_conforming_class_is_instance(self) -> None:
        class _GoodTool:
            name: str = "good"
            description: str = "A good tool"

            async def execute(self, input_text: str) -> ToolResult:
                return ToolResult(output=input_text)

        assert isinstance(_GoodTool(), Tool)

    def test_missing_execute_is_not_instance(self) -> None:
        class _BadTool:
            name: str = "bad"
            description: str = "Missing execute"

        assert not isinstance(_BadTool(), Tool)

    def test_missing_name_is_not_instance(self) -> None:
        class _NoName:
            description: str = "No name"

            async def execute(self, input_text: str) -> ToolResult:
                return ToolResult(output=input_text)

        assert not isinstance(_NoName(), Tool)

    def test_missing_description_is_not_instance(self) -> None:
        class _NoDesc:
            name: str = "no_desc"

            async def execute(self, input_text: str) -> ToolResult:
                return ToolResult(output=input_text)

        assert not isinstance(_NoDesc(), Tool)

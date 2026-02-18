"""FallbackStrategy のユニットテスト。

ツール実行失敗時のリトライと degradation を検証する。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.agent.fallback import FallbackResult, FallbackStrategy
from src.agent.tools.base import Tool, ToolResult

# --- テスト用ツール ---


class SuccessTool:
    """常に成功するツール。"""

    def __init__(self) -> None:
        self.name = "success_tool"
        self.description = "Always succeeds."

    async def execute(self, input_text: str) -> ToolResult:
        return ToolResult(output="success")


class ErrorTool:
    """常にエラーを返すツール。"""

    def __init__(self) -> None:
        self.name = "error_tool"
        self.description = "Always returns error."

    async def execute(self, input_text: str) -> ToolResult:
        return ToolResult(output="", error="tool error")


class ExceptionTool:
    """常に例外を送出するツール。"""

    def __init__(self) -> None:
        self.name = "exception_tool"
        self.description = "Always raises exception."

    async def execute(self, input_text: str) -> ToolResult:
        raise RuntimeError("unexpected failure")


# --- FallbackResult ---


class TestFallbackResult:
    """FallbackResult モデルのテスト。"""

    def test_result_success(self) -> None:
        result = FallbackResult(
            tool_result=ToolResult(output="ok"),
            retries_attempted=0,
            degraded=False,
        )
        assert result.tool_result.output == "ok"
        assert result.retries_attempted == 0
        assert result.degraded is False

    def test_result_degraded(self) -> None:
        result = FallbackResult(
            tool_result=ToolResult(output="", error="all failed"),
            retries_attempted=2,
            degraded=True,
        )
        assert result.degraded is True
        assert result.retries_attempted == 2


# --- FallbackStrategy: 成功ケース ---


class TestFallbackStrategySuccess:
    """ツールが成功する場合のテスト。"""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self) -> None:
        strategy = FallbackStrategy(max_retries=2)
        tool = SuccessTool()
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.tool_result.output == "success"
        assert result.tool_result.is_error is False
        assert result.retries_attempted == 0
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_zero_max_retries_succeeds_on_first_try(self) -> None:
        strategy = FallbackStrategy(max_retries=0)
        tool = SuccessTool()
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.tool_result.output == "success"
        assert result.retries_attempted == 0
        assert result.degraded is False


# --- FallbackStrategy: リトライケース ---


class TestFallbackStrategyRetry:
    """リトライのテスト。"""

    @pytest.mark.asyncio
    async def test_retries_on_tool_result_error(self) -> None:
        """ToolResult.is_error == True の場合にリトライする。"""
        tool = AsyncMock(spec=Tool)
        tool.name = "mock_tool"
        tool.description = "Mock tool"
        tool.execute = AsyncMock(
            side_effect=[
                ToolResult(output="", error="transient error"),
                ToolResult(output="recovered"),
            ]
        )
        strategy = FallbackStrategy(max_retries=2)
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.tool_result.output == "recovered"
        assert result.retries_attempted == 1
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_retries_on_exception(self) -> None:
        """例外発生時にリトライする。"""
        tool = AsyncMock(spec=Tool)
        tool.name = "mock_tool"
        tool.description = "Mock tool"
        tool.execute = AsyncMock(
            side_effect=[
                RuntimeError("boom"),
                ToolResult(output="recovered after exception"),
            ]
        )
        strategy = FallbackStrategy(max_retries=2)
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.tool_result.output == "recovered after exception"
        assert result.retries_attempted == 1
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_retries_exact_max_times(self) -> None:
        """最大リトライ回数ちょうどで成功する。"""
        tool = AsyncMock(spec=Tool)
        tool.name = "mock_tool"
        tool.description = "Mock tool"
        tool.execute = AsyncMock(
            side_effect=[
                ToolResult(output="", error="fail 1"),
                ToolResult(output="", error="fail 2"),
                ToolResult(output="success on third"),
            ]
        )
        strategy = FallbackStrategy(max_retries=2)
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.tool_result.output == "success on third"
        assert result.retries_attempted == 2
        assert result.degraded is False


# --- FallbackStrategy: 全試行失敗 → degraded ---


class TestFallbackStrategyDegraded:
    """全試行失敗時の degradation テスト。"""

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_degraded(self) -> None:
        strategy = FallbackStrategy(max_retries=2)
        tool = ErrorTool()
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.degraded is True
        assert result.retries_attempted == 2
        assert result.tool_result.is_error is True

    @pytest.mark.asyncio
    async def test_all_exceptions_exhausted_returns_degraded(self) -> None:
        strategy = FallbackStrategy(max_retries=1)
        tool = ExceptionTool()
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.degraded is True
        assert result.retries_attempted == 1
        assert result.tool_result.is_error is True
        assert result.tool_result.error is not None

    @pytest.mark.asyncio
    async def test_zero_retries_failure_returns_degraded(self) -> None:
        """max_retries=0 で初回失敗時は即座に degraded。"""
        strategy = FallbackStrategy(max_retries=0)
        tool = ErrorTool()
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.degraded is True
        assert result.retries_attempted == 0
        assert result.tool_result.is_error is True

    @pytest.mark.asyncio
    async def test_exception_converted_to_tool_result_error(self) -> None:
        """例外が ToolResult(error=...) に変換されること。"""
        strategy = FallbackStrategy(max_retries=0)
        tool = ExceptionTool()
        result = await strategy.execute_with_fallback(tool, "input")
        assert result.tool_result.is_error is True
        assert "unexpected failure" in (result.tool_result.error or "")


# --- FallbackStrategy: デフォルト設定 ---


class TestFallbackStrategyDefaults:
    """デフォルト設定のテスト。"""

    def test_default_max_retries_is_two(self) -> None:
        strategy = FallbackStrategy()
        assert strategy._max_retries == 2

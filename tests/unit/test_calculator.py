"""CalculatorTool のユニットテスト。

四則演算、べき乗、剰余、エラー処理、安全性を検証する。
"""

from __future__ import annotations

import pytest
from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.calculator import CalculatorTool


# ---------------------------------------------------------------------------
# TestCalculatorToolAttributes
# ---------------------------------------------------------------------------
class TestCalculatorToolAttributes:
    """CalculatorTool の属性と Protocol 準拠を検証する。"""

    def test_name(self) -> None:
        tool = CalculatorTool()
        assert tool.name == "calculator"

    def test_description_is_non_empty(self) -> None:
        tool = CalculatorTool()
        assert len(tool.description) > 0

    def test_conforms_to_tool_protocol(self) -> None:
        tool = CalculatorTool()
        assert isinstance(tool, Tool)


# ---------------------------------------------------------------------------
# TestCalculatorToolBasicOperations
# ---------------------------------------------------------------------------
class TestCalculatorToolBasicOperations:
    """基本的な算術演算を検証する。"""

    @pytest.mark.asyncio
    async def test_addition(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("2 + 3")
        assert result.output == "5"

    @pytest.mark.asyncio
    async def test_subtraction(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("10 - 4")
        assert result.output == "6"

    @pytest.mark.asyncio
    async def test_multiplication(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("3 * 7")
        assert result.output == "21"

    @pytest.mark.asyncio
    async def test_division(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("15 / 4")
        assert result.output == "3.75"

    @pytest.mark.asyncio
    async def test_integer_division(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("15 // 4")
        assert result.output == "3"

    @pytest.mark.asyncio
    async def test_modulo(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("17 % 5")
        assert result.output == "2"

    @pytest.mark.asyncio
    async def test_power(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("2 ** 10")
        assert result.output == "1024"

    @pytest.mark.asyncio
    async def test_unary_negative(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("-5 + 3")
        assert result.output == "-2"

    @pytest.mark.asyncio
    async def test_unary_positive(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("+5")
        assert result.output == "5"

    @pytest.mark.asyncio
    async def test_complex_expression(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("(2 + 3) * 4 - 1")
        assert result.output == "19"

    @pytest.mark.asyncio
    async def test_float_arithmetic(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("0.1 + 0.2")
        # 浮動小数点の結果を数値として比較
        assert abs(float(result.output) - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# TestCalculatorToolResult
# ---------------------------------------------------------------------------
class TestCalculatorToolResult:
    """戻り値の型とフィールドを検証する。"""

    @pytest.mark.asyncio
    async def test_returns_tool_result(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("1 + 1")
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_success_has_no_error(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("1 + 1")
        assert result.error is None
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_output_is_string(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("2 + 3")
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_integer_result_has_no_decimal(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("2 + 3")
        assert "." not in result.output


# ---------------------------------------------------------------------------
# TestCalculatorToolErrorHandling
# ---------------------------------------------------------------------------
class TestCalculatorToolErrorHandling:
    """エラー処理を検証する。CalculatorTool は例外を raise せず ToolResult で返す。"""

    @pytest.mark.asyncio
    async def test_division_by_zero(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("1 / 0")
        assert result.is_error is True
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_invalid_expression(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("not a math expression")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_error_result_has_empty_output(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("invalid")
        assert result.output == ""


# ---------------------------------------------------------------------------
# TestCalculatorToolSecurity
# ---------------------------------------------------------------------------
class TestCalculatorToolSecurity:
    """安全性を検証する。関数呼び出し、import、属性アクセスを拒否する。"""

    @pytest.mark.asyncio
    async def test_rejects_function_call(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("__import__('os').system('ls')")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_rejects_builtin_function(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("print('hello')")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_rejects_attribute_access(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("(1).__class__")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_rejects_import(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("import os")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_rejects_name_reference(self) -> None:
        tool = CalculatorTool()
        result = await tool.execute("x + 1")
        assert result.is_error is True

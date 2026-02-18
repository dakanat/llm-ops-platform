"""計算ツール。

ast モジュールで数式を安全に評価する。
"""

from __future__ import annotations

import ast
import operator
from typing import Any, ClassVar

from src.agent.tools.base import ToolResult

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> int | float:
    """AST ノードを再帰的に評価する。

    許可されたノードのみ処理し、それ以外は ValueError を送出する。

    Args:
        node: 評価する AST ノード。

    Returns:
        計算結果。

    Raises:
        ValueError: 許可されていないノードの場合。
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left: int | float = _eval_node(node.left)
        right: int | float = _eval_node(node.right)
        result: int | float = op_func(left, right)
        return result

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand: int | float = _eval_node(node.operand)
        result = op_func(operand)
        return result

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


class CalculatorTool:
    """安全な数式評価ツール。

    Python の ast モジュールで式をパースし、許可された演算のみを実行する。
    関数呼び出し、属性アクセス、import 等は AST レベルで拒否する。
    """

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = "Evaluate a mathematical expression safely."

    async def execute(self, input_text: str) -> ToolResult:
        """数式を評価して結果を返す。

        エラー時は例外を raise せず ToolResult.error に格納する。

        Args:
            input_text: 評価する数式文字列。

        Returns:
            計算結果を含む ToolResult。
        """
        if not input_text.strip():
            return ToolResult(output="", error="Empty expression")

        try:
            tree = ast.parse(input_text.strip(), mode="eval")
        except SyntaxError as e:
            return ToolResult(output="", error=f"Invalid expression: {e}")

        try:
            value = _eval_node(tree)
        except (ValueError, ZeroDivisionError) as e:
            return ToolResult(output="", error=str(e))

        # 整数結果は小数点なしで表示
        if isinstance(value, float) and value == int(value):
            value = int(value)

        return ToolResult(output=str(value))

"""データベースクエリツール。

読み取り専用の SELECT クエリのみを許可し、SQL インジェクションを防止する。
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.tools.base import ToolResult

# SELECT 以外のステートメントおよび危険なパターンを検出する正規表現
_FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r";\s*", re.IGNORECASE),  # 複数ステートメント
    re.compile(r"--", re.IGNORECASE),  # ラインコメント
    re.compile(r"/\*", re.IGNORECASE),  # ブロックコメント
    re.compile(r"\bINTO\b", re.IGNORECASE),  # SELECT INTO
]

_WRITE_KEYWORDS: set[str] = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
}


def _validate_query(sql: str) -> str | None:
    """SQL クエリを検証し、安全でない場合はエラーメッセージを返す。

    Args:
        sql: 検証する SQL 文字列。

    Returns:
        エラーメッセージ。安全な場合は None。
    """
    stripped = sql.strip()

    if not stripped:
        return "Empty query"

    # SELECT で始まることを確認
    if not stripped.upper().startswith("SELECT"):
        return "Only SELECT queries are allowed"

    # 書き込み系キーワードの検出
    upper = stripped.upper()
    for keyword in _WRITE_KEYWORDS:
        pattern = re.compile(rf"\b{keyword}\b", re.IGNORECASE)
        if pattern.search(upper):
            return f"Forbidden keyword: {keyword}"

    # 危険なパターンの検出
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(stripped):
            return "Query contains forbidden pattern"

    return None


class DatabaseTool:
    """読み取り専用データベースクエリツール。

    SELECT クエリのみを許可し、書き込み・DDL・複数ステートメントを拒否する。
    結果は JSON 形式で返す。
    """

    name: ClassVar[str] = "database"
    description: ClassVar[str] = (
        "Execute a read-only SQL query against the database. "
        "Only SELECT statements are allowed. "
        "Input is a SQL SELECT query string."
    )

    def __init__(self, session: AsyncSession, max_rows: int = 100) -> None:
        """DatabaseTool を初期化する。

        Args:
            session: 非同期データベースセッション。
            max_rows: 返却する最大行数。デフォルトは 100。
        """
        self._session = session
        self._max_rows = max_rows

    async def execute(self, input_text: str) -> ToolResult:
        """SQL クエリを実行して結果を返す。

        エラー時は例外を raise せず ToolResult.error に格納する。

        Args:
            input_text: 実行する SQL クエリ文字列。

        Returns:
            クエリ結果を JSON 形式で含む ToolResult。
        """
        if not input_text.strip():
            return ToolResult(output="", error="Empty query")

        validation_error = _validate_query(input_text)
        if validation_error is not None:
            return ToolResult(output="", error=validation_error)

        try:
            result = await self._session.execute(text(input_text))
            rows: list[dict[str, Any]] = [dict(row) for row in result.mappings().all()]
        except Exception as e:
            return ToolResult(output="", error=str(e))

        # 行数制限
        truncated = len(rows) > self._max_rows
        rows = rows[: self._max_rows]

        return ToolResult(output=self._format_output(rows, truncated))

    @staticmethod
    def _format_output(rows: list[dict[str, Any]], truncated: bool) -> str:
        """クエリ結果を可読な JSON 文字列に変換する。

        Args:
            rows: クエリ結果の辞書リスト。
            truncated: 行数制限で切り詰められた場合 True。

        Returns:
            行数情報と JSON データを含むフォーマット済み文字列。
        """
        count = len(rows)
        parts: list[str] = []

        if truncated:
            parts.append(f"Showing {count} rows (truncated):")
        else:
            parts.append(f"{count} rows returned:")

        parts.append(json.dumps(rows, ensure_ascii=False, default=str))

        return "\n".join(parts)

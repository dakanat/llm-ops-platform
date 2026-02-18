"""DatabaseTool のユニットテスト。

DB クエリツールが安全にクエリを実行し、SQL インジェクションを防止することを検証する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from typing import Any

import pytest
from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.database import DatabaseTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_session_mock(
    rows: list[dict[str, Any]] | None = None,
    error: Exception | None = None,
) -> AsyncMock:
    """AsyncSession のモックを生成する。

    execute() が返す Result オブジェクトのモックを構築する。
    """
    session = AsyncMock()
    if error is not None:
        session.execute.side_effect = error
    else:
        result_mock = MagicMock()
        if rows is not None:
            # mappings().all() で辞書のリストを返す
            result_mock.mappings.return_value.all.return_value = rows
            result_mock.keys.return_value = list(rows[0].keys()) if rows else []
        else:
            result_mock.mappings.return_value.all.return_value = []
            result_mock.keys.return_value = []
        session.execute.return_value = result_mock
    return session


# ---------------------------------------------------------------------------
# TestDatabaseToolAttributes
# ---------------------------------------------------------------------------
class TestDatabaseToolAttributes:
    """DatabaseTool の属性と Protocol 準拠を検証する。"""

    def test_name(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        assert tool.name == "database"

    def test_description_is_non_empty(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        assert len(tool.description) > 0

    def test_conforms_to_tool_protocol(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        assert isinstance(tool, Tool)


# ---------------------------------------------------------------------------
# TestDatabaseToolExecute
# ---------------------------------------------------------------------------
class TestDatabaseToolExecute:
    """DatabaseTool の正常系実行を検証する。"""

    @pytest.mark.asyncio
    async def test_returns_tool_result(self) -> None:
        session = _make_session_mock(rows=[])
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT 1")
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_success_has_no_error(self) -> None:
        session = _make_session_mock(rows=[])
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT 1")
        assert result.error is None
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_returns_rows_as_json(self) -> None:
        rows = [
            {"id": 1, "title": "ドキュメント1"},
            {"id": 2, "title": "ドキュメント2"},
        ]
        session = _make_session_mock(rows=rows)
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT id, title FROM documents")
        assert "ドキュメント1" in result.output
        assert "ドキュメント2" in result.output

    @pytest.mark.asyncio
    async def test_empty_result_returns_no_error(self) -> None:
        session = _make_session_mock(rows=[])
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT * FROM documents WHERE id = 'none'")
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_result_count_in_output(self) -> None:
        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        session = _make_session_mock(rows=rows)
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT id FROM documents")
        assert "3" in result.output


# ---------------------------------------------------------------------------
# TestDatabaseToolSecurity
# ---------------------------------------------------------------------------
class TestDatabaseToolSecurity:
    """SQL インジェクション防止を検証する。SELECT 以外のステートメントを拒否する。"""

    @pytest.mark.asyncio
    async def test_rejects_drop_table(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("DROP TABLE users")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_delete(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("DELETE FROM users WHERE id = 1")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_insert(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("INSERT INTO users (name) VALUES ('evil')")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_update(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("UPDATE users SET role = 'admin'")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_alter(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("ALTER TABLE users ADD COLUMN evil TEXT")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_truncate(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("TRUNCATE TABLE users")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_create(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("CREATE TABLE evil (id INT)")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_semicolon_multiple_statements(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT 1; DROP TABLE users")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_comment_injection(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT 1 -- DROP TABLE users")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_grant(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("GRANT ALL ON users TO evil")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_select_case_insensitive(self) -> None:
        session = _make_session_mock(rows=[])
        tool = DatabaseTool(session=session)
        result = await tool.execute("select * from documents")
        assert result.is_error is False
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_select_into(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT * INTO new_table FROM users")
        assert result.is_error is True
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_select_with_where(self) -> None:
        session = _make_session_mock(rows=[{"id": 1}])
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT id FROM documents WHERE title = 'test'")
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_allows_select_with_join(self) -> None:
        session = _make_session_mock(rows=[])
        tool = DatabaseTool(session=session)
        result = await tool.execute(
            "SELECT d.title FROM documents d JOIN chunks c ON d.id = c.document_id"
        )
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_allows_select_count(self) -> None:
        session = _make_session_mock(rows=[{"count": 5}])
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT COUNT(*) FROM documents")
        assert result.is_error is False


# ---------------------------------------------------------------------------
# TestDatabaseToolErrorHandling
# ---------------------------------------------------------------------------
class TestDatabaseToolErrorHandling:
    """エラー処理を検証する。例外を raise せず ToolResult で返す。"""

    @pytest.mark.asyncio
    async def test_db_error_returns_error_result(self) -> None:
        session = _make_session_mock(error=Exception("connection refused"))
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT 1")
        assert result.is_error is True
        assert "connection refused" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_db_error_has_empty_output(self) -> None:
        session = _make_session_mock(error=Exception("DB error"))
        tool = DatabaseTool(session=session)
        result = await tool.execute("SELECT 1")
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_empty_input_returns_error(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_whitespace_only_input_returns_error(self) -> None:
        session = _make_session_mock()
        tool = DatabaseTool(session=session)
        result = await tool.execute("   ")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_max_rows_limits_result(self) -> None:
        rows = [{"id": i} for i in range(200)]
        session = _make_session_mock(rows=rows)
        tool = DatabaseTool(session=session, max_rows=100)
        result = await tool.execute("SELECT id FROM documents")
        assert result.is_error is False
        assert "100" in result.output

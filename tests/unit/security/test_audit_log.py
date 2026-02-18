"""監査ログのユニットテスト。

create_audit_log による AuditLog 生成・DB 追加、
log_action による構造化ログ連携を検証する。
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.db.models import AuditLog
from src.security.audit_log import create_audit_log, log_action

# --- create_audit_log ---


class TestCreateAuditLog:
    """create_audit_log のテスト。"""

    @pytest.mark.asyncio
    async def test_returns_audit_log_instance(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()
        result = await create_audit_log(
            session=session,
            user_id=user_id,
            action="login",
            resource_type="auth",
            resource_id="session-123",
        )
        assert isinstance(result, AuditLog)

    @pytest.mark.asyncio
    async def test_fields_are_set_correctly(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()
        result = await create_audit_log(
            session=session,
            user_id=user_id,
            action="create",
            resource_type="document",
            resource_id="doc-456",
            details={"title": "Test Document"},
        )
        assert result.user_id == user_id
        assert result.action == "create"
        assert result.resource_type == "document"
        assert result.resource_id == "doc-456"
        assert result.details == {"title": "Test Document"}

    @pytest.mark.asyncio
    async def test_default_details_is_empty_dict(self) -> None:
        session = AsyncMock()
        result = await create_audit_log(
            session=session,
            user_id=uuid.uuid4(),
            action="read",
            resource_type="document",
            resource_id="doc-789",
        )
        assert result.details == {}

    @pytest.mark.asyncio
    async def test_session_add_called(self) -> None:
        session = AsyncMock()
        await create_audit_log(
            session=session,
            user_id=uuid.uuid4(),
            action="delete",
            resource_type="document",
            resource_id="doc-000",
        )
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_flush_called(self) -> None:
        session = AsyncMock()
        await create_audit_log(
            session=session,
            user_id=uuid.uuid4(),
            action="update",
            resource_type="user",
            resource_id="user-111",
        )
        session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_commit_not_called(self) -> None:
        """create_audit_log は commit しない (呼び出し元が制御)。"""
        session = AsyncMock()
        await create_audit_log(
            session=session,
            user_id=uuid.uuid4(),
            action="read",
            resource_type="document",
            resource_id="doc-222",
        )
        session.commit.assert_not_awaited()


# --- log_action ---


class TestLogAction:
    """log_action のテスト。"""

    @pytest.mark.asyncio
    async def test_returns_audit_log(self) -> None:
        session = AsyncMock()
        result = await log_action(
            session=session,
            user_id=uuid.uuid4(),
            action="login",
            resource_type="auth",
            resource_id="session-123",
        )
        assert isinstance(result, AuditLog)

    @pytest.mark.asyncio
    async def test_emits_structlog_event(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        with patch("src.security.audit_log.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            await log_action(
                session=session,
                user_id=user_id,
                action="create",
                resource_type="document",
                resource_id="doc-456",
                details={"key": "value"},
            )

            mock_logger.info.assert_called_once_with(
                "audit_action",
                user_id=str(user_id),
                action="create",
                resource_type="document",
                resource_id="doc-456",
                details={"key": "value"},
            )

    @pytest.mark.asyncio
    async def test_delegates_to_create_audit_log(self) -> None:
        session = AsyncMock()
        user_id = uuid.uuid4()

        with patch(
            "src.security.audit_log.create_audit_log", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = AuditLog(
                user_id=user_id,
                action="read",
                resource_type="doc",
                resource_id="d-1",
            )
            result = await log_action(
                session=session,
                user_id=user_id,
                action="read",
                resource_type="doc",
                resource_id="d-1",
            )
            mock_create.assert_awaited_once_with(
                session=session,
                user_id=user_id,
                action="read",
                resource_type="doc",
                resource_id="d-1",
                details=None,
            )
            assert result.action == "read"

"""監査ログモジュール。

AuditLog レコードの作成と構造化ログへの出力を提供する。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.models import AuditLog
from src.monitoring.logger import get_logger


async def create_audit_log(
    *,
    session: AsyncSession,
    user_id: UUID,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    """監査ログレコードを作成して DB に追加する。

    commit は呼び出し元に委ねる (flush のみ実行)。

    Args:
        session: 非同期 DB セッション。
        user_id: 操作実行ユーザーの ID。
        action: 実行されたアクション (例: "create", "delete")。
        resource_type: 対象リソース種別 (例: "document", "user")。
        resource_id: 対象リソースの識別子。
        details: 追加情報の辞書 (省略時は空辞書)。

    Returns:
        作成された AuditLog インスタンス。
    """
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details if details is not None else {},
    )
    session.add(audit_log)
    await session.flush()
    return audit_log


async def log_action(
    *,
    session: AsyncSession,
    user_id: UUID,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    """監査ログレコードを作成し、構造化ログにもイベントを出力する。

    Args:
        session: 非同期 DB セッション。
        user_id: 操作実行ユーザーの ID。
        action: 実行されたアクション。
        resource_type: 対象リソース種別。
        resource_id: 対象リソースの識別子。
        details: 追加情報の辞書。

    Returns:
        作成された AuditLog インスタンス。
    """
    audit_log = await create_audit_log(
        session=session,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    )
    logger = get_logger()
    logger.info(
        "audit_action",
        user_id=str(user_id),
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    )
    return audit_log

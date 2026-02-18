"""RBAC権限管理モジュール。

Role / Permission enum、ロール別権限マトリクス、
has_permission / has_role 純粋関数、
require_permission / require_role FastAPI 依存関数を提供する。
"""

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status


class Role(StrEnum):
    """ユーザーロール。"""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class Permission(StrEnum):
    """操作権限。"""

    CHAT = "chat"
    RAG_QUERY = "rag:query"
    RAG_INDEX = "rag:index"
    AGENT_RUN = "agent:run"
    EVAL_RUN = "eval:run"
    EVAL_READ = "eval:read"
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    USER_MANAGE = "user:manage"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),
    Role.USER: {
        Permission.CHAT,
        Permission.RAG_QUERY,
        Permission.RAG_INDEX,
        Permission.AGENT_RUN,
        Permission.EVAL_RUN,
        Permission.EVAL_READ,
    },
    Role.VIEWER: {
        Permission.CHAT,
        Permission.RAG_QUERY,
        Permission.EVAL_READ,
    },
}

_ROLE_HIERARCHY: dict[str, int] = {
    Role.VIEWER: 0,
    Role.USER: 1,
    Role.ADMIN: 2,
}


def has_permission(role: str, permission: Permission) -> bool:
    """指定ロールが指定権限を持つか判定する。

    Args:
        role: ユーザーロール文字列。
        permission: チェック対象の権限。

    Returns:
        権限がある場合 True、不明なロールの場合 False。
    """
    try:
        role_enum = Role(role)
    except ValueError:
        return False
    return permission in ROLE_PERMISSIONS[role_enum]


def has_role(role: str, required_role: str) -> bool:
    """ロール階層に基づき、必要ロール以上かを判定する。

    Args:
        role: ユーザーの現在ロール。
        required_role: 要求されるロール。

    Returns:
        階層的に十分な場合 True。
    """
    user_level = _ROLE_HIERARCHY.get(role)
    required_level = _ROLE_HIERARCHY.get(required_role)
    if user_level is None or required_level is None:
        return False
    return user_level >= required_level


def require_permission(permission: Permission) -> Callable[..., Any]:
    """指定権限を要求する FastAPI 依存関数を返すファクトリ。

    Args:
        permission: 必要な権限。

    Returns:
        FastAPI Depends で使用可能な依存関数。
    """
    from src.api.middleware.auth import TokenPayload, get_current_user

    async def _check(
        current_user: Annotated[TokenPayload, Depends(get_current_user)],
    ) -> TokenPayload:
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return current_user

    return _check


def require_role(required_role: str) -> Callable[..., Any]:
    """指定ロール以上を要求する FastAPI 依存関数を返すファクトリ。

    Args:
        required_role: 必要な最低ロール。

    Returns:
        FastAPI Depends で使用可能な依存関数。
    """
    from src.api.middleware.auth import TokenPayload, get_current_user

    async def _check(
        current_user: Annotated[TokenPayload, Depends(get_current_user)],
    ) -> TokenPayload:
        if not has_role(current_user.role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' or higher required",
            )
        return current_user

    return _check

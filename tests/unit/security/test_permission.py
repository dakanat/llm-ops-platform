"""RBAC権限管理のユニットテスト。

Role/Permission enum、権限マトリクス、has_permission/has_role 純粋関数、
require_permission/require_role FastAPI 依存関数を検証する。
"""

import uuid
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from src.api.middleware.auth import TokenPayload, create_access_token
from src.security.permission import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    has_permission,
    has_role,
    require_permission,
    require_role,
)

# --- Role ---


class TestRole:
    """Role enum のテスト。"""

    def test_has_expected_members(self) -> None:
        assert set(Role) == {Role.ADMIN, Role.USER, Role.VIEWER}

    def test_values_are_strings(self) -> None:
        assert Role.ADMIN.value == "admin"
        assert Role.USER.value == "user"
        assert Role.VIEWER.value == "viewer"

    def test_role_is_str_enum(self) -> None:
        assert isinstance(Role.ADMIN, str)
        assert str(Role.ADMIN) == "admin"


# --- Permission ---


class TestPermission:
    """Permission enum のテスト。"""

    def test_has_expected_members(self) -> None:
        assert set(Permission) == {
            Permission.CHAT,
            Permission.RAG_QUERY,
            Permission.RAG_INDEX,
            Permission.AGENT_RUN,
            Permission.EVAL_RUN,
            Permission.EVAL_READ,
            Permission.ADMIN_READ,
            Permission.ADMIN_WRITE,
            Permission.USER_MANAGE,
        }

    def test_values_are_strings(self) -> None:
        assert Permission.CHAT.value == "chat"
        assert Permission.RAG_QUERY.value == "rag:query"
        assert Permission.RAG_INDEX.value == "rag:index"
        assert Permission.AGENT_RUN.value == "agent:run"
        assert Permission.EVAL_RUN.value == "eval:run"
        assert Permission.EVAL_READ.value == "eval:read"
        assert Permission.ADMIN_READ.value == "admin:read"
        assert Permission.ADMIN_WRITE.value == "admin:write"
        assert Permission.USER_MANAGE.value == "user:manage"


# --- ROLE_PERMISSIONS ---


class TestRolePermissions:
    """権限マトリクスのテスト。"""

    def test_admin_has_all_permissions(self) -> None:
        assert ROLE_PERMISSIONS[Role.ADMIN] == set(Permission)

    def test_user_is_subset_of_admin(self) -> None:
        assert ROLE_PERMISSIONS[Role.USER] < ROLE_PERMISSIONS[Role.ADMIN]

    def test_viewer_is_subset_of_user(self) -> None:
        assert ROLE_PERMISSIONS[Role.VIEWER] < ROLE_PERMISSIONS[Role.USER]

    def test_viewer_has_no_write_permissions(self) -> None:
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.RAG_INDEX not in viewer_perms
        assert Permission.ADMIN_WRITE not in viewer_perms
        assert Permission.USER_MANAGE not in viewer_perms
        assert Permission.EVAL_RUN not in viewer_perms
        assert Permission.AGENT_RUN not in viewer_perms

    def test_viewer_has_read_permissions(self) -> None:
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.CHAT in viewer_perms
        assert Permission.RAG_QUERY in viewer_perms
        assert Permission.EVAL_READ in viewer_perms

    def test_user_has_operational_permissions(self) -> None:
        user_perms = ROLE_PERMISSIONS[Role.USER]
        assert Permission.CHAT in user_perms
        assert Permission.RAG_QUERY in user_perms
        assert Permission.RAG_INDEX in user_perms
        assert Permission.AGENT_RUN in user_perms
        assert Permission.EVAL_RUN in user_perms
        assert Permission.EVAL_READ in user_perms

    def test_user_has_no_admin_permissions(self) -> None:
        user_perms = ROLE_PERMISSIONS[Role.USER]
        assert Permission.ADMIN_READ not in user_perms
        assert Permission.ADMIN_WRITE not in user_perms
        assert Permission.USER_MANAGE not in user_perms


# --- has_permission ---


class TestHasPermission:
    """has_permission 関数のテスト。"""

    def test_admin_has_all_permissions(self) -> None:
        for perm in Permission:
            assert has_permission("admin", perm) is True

    def test_user_has_chat(self) -> None:
        assert has_permission("user", Permission.CHAT) is True

    def test_user_has_no_admin_write(self) -> None:
        assert has_permission("user", Permission.ADMIN_WRITE) is False

    def test_viewer_has_chat(self) -> None:
        assert has_permission("viewer", Permission.CHAT) is True

    def test_viewer_has_no_rag_index(self) -> None:
        assert has_permission("viewer", Permission.RAG_INDEX) is False

    def test_unknown_role_denies_all(self) -> None:
        for perm in Permission:
            assert has_permission("unknown", perm) is False


# --- has_role ---


class TestHasRole:
    """has_role 関数のテスト。"""

    def test_admin_has_admin_role(self) -> None:
        assert has_role("admin", "admin") is True

    def test_admin_has_user_role(self) -> None:
        assert has_role("admin", "user") is True

    def test_admin_has_viewer_role(self) -> None:
        assert has_role("admin", "viewer") is True

    def test_user_has_user_role(self) -> None:
        assert has_role("user", "user") is True

    def test_user_has_viewer_role(self) -> None:
        assert has_role("user", "viewer") is True

    def test_user_does_not_have_admin_role(self) -> None:
        assert has_role("user", "admin") is False

    def test_viewer_has_viewer_role(self) -> None:
        assert has_role("viewer", "viewer") is True

    def test_viewer_does_not_have_user_role(self) -> None:
        assert has_role("viewer", "user") is False

    def test_viewer_does_not_have_admin_role(self) -> None:
        assert has_role("viewer", "admin") is False

    def test_unknown_role_denies_all(self) -> None:
        assert has_role("unknown", "viewer") is False
        assert has_role("unknown", "user") is False
        assert has_role("unknown", "admin") is False


# --- require_permission (FastAPI DI) ---


def _make_permission_app(permission: Permission) -> FastAPI:
    """テスト用 FastAPI アプリを生成する。"""
    app = FastAPI()

    @app.get("/protected")
    async def protected(
        _user: Annotated[TokenPayload, Depends(require_permission(permission))],
    ) -> dict[str, str]:
        return {"status": "ok"}

    return app


class TestRequirePermission:
    """require_permission DI ファクトリのテスト。"""

    def test_admin_can_access_admin_write(self) -> None:
        app = _make_permission_app(Permission.ADMIN_WRITE)
        client = TestClient(app)
        token = create_access_token(uuid.uuid4(), "admin@example.com", "admin")
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_user_denied_admin_write(self) -> None:
        app = _make_permission_app(Permission.ADMIN_WRITE)
        client = TestClient(app)
        token = create_access_token(uuid.uuid4(), "user@example.com", "user")
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_no_token_returns_401(self) -> None:
        app = _make_permission_app(Permission.CHAT)
        client = TestClient(app)
        resp = client.get("/protected")
        assert resp.status_code == 401


# --- require_role (FastAPI DI) ---


def _make_role_app(required_role: str) -> FastAPI:
    """テスト用 FastAPI アプリを生成する。"""
    app = FastAPI()

    @app.get("/protected")
    async def protected(
        _user: Annotated[TokenPayload, Depends(require_role(required_role))],
    ) -> dict[str, str]:
        return {"status": "ok"}

    return app


class TestRequireRole:
    """require_role DI ファクトリのテスト。"""

    def test_admin_can_access_admin_required(self) -> None:
        app = _make_role_app("admin")
        client = TestClient(app)
        token = create_access_token(uuid.uuid4(), "admin@example.com", "admin")
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_user_denied_admin_required(self) -> None:
        app = _make_role_app("admin")
        client = TestClient(app)
        token = create_access_token(uuid.uuid4(), "user@example.com", "user")
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_user_can_access_viewer_required(self) -> None:
        app = _make_role_app("viewer")
        client = TestClient(app)
        token = create_access_token(uuid.uuid4(), "user@example.com", "user")
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_no_token_returns_401(self) -> None:
        app = _make_role_app("viewer")
        client = TestClient(app)
        resp = client.get("/protected")
        assert resp.status_code == 401

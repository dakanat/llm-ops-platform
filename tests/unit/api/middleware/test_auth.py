"""JWT認証・パスワードハッシュのユニットテスト。

bcrypt パスワードハッシュ、JWT トークン生成・検証、
get_current_user FastAPI 依存関数を検証する。
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from src.api.middleware.auth import (
    TokenPayload,
    create_access_token,
    decode_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from src.config import Settings

# --- Password Hashing ---


class TestPasswordHashing:
    """bcrypt パスワードハッシュのテスト。"""

    def test_hash_returns_string(self) -> None:
        hashed = hash_password("mypassword")
        assert isinstance(hashed, str)

    def test_hash_is_not_plaintext(self) -> None:
        hashed = hash_password("mypassword")
        assert hashed != "mypassword"

    def test_hash_is_salted(self) -> None:
        """同じパスワードでも毎回異なるハッシュが生成される。"""
        h1 = hash_password("mypassword")
        h2 = hash_password("mypassword")
        assert h1 != h2

    def test_verify_correct_password(self) -> None:
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed) is True

    def test_verify_wrong_password(self) -> None:
        hashed = hash_password("mypassword")
        assert verify_password("wrongpassword", hashed) is False


# --- TokenPayload ---


class TestTokenPayload:
    """TokenPayload Pydantic モデルのテスト。"""

    def test_fields(self) -> None:
        user_id = uuid.uuid4()
        payload = TokenPayload(
            sub=str(user_id),
            email="user@example.com",
            role="admin",
            exp=datetime.now(UTC),
        )
        assert payload.sub == str(user_id)
        assert payload.email == "user@example.com"
        assert payload.role == "admin"
        assert isinstance(payload.exp, datetime)

    def test_required_fields(self) -> None:
        fields = set(TokenPayload.model_fields.keys())
        assert {"sub", "email", "role", "exp"} == fields


# --- create_access_token ---


class TestCreateAccessToken:
    """JWT トークン生成のテスト。"""

    def test_returns_string(self) -> None:
        token = create_access_token(uuid.uuid4(), "user@example.com", "user")
        assert isinstance(token, str)

    def test_token_contains_correct_claims(self) -> None:
        user_id = uuid.uuid4()
        settings = Settings()
        token = create_access_token(user_id, "admin@example.com", "admin", settings)
        decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        assert decoded["sub"] == str(user_id)
        assert decoded["email"] == "admin@example.com"
        assert decoded["role"] == "admin"
        assert "exp" in decoded

    def test_exp_is_in_future(self) -> None:
        settings = Settings()
        token = create_access_token(uuid.uuid4(), "u@e.com", "user", settings)
        decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        exp = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        assert exp > datetime.now(UTC)

    def test_custom_settings_applied(self) -> None:
        settings = Settings(jwt_access_token_expire_minutes=60)
        token = create_access_token(uuid.uuid4(), "u@e.com", "user", settings)
        decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        exp = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        # 有効期限が55分以上先であること (マージン5分)
        assert exp > datetime.now(UTC) + timedelta(minutes=55)


# --- decode_access_token ---


class TestDecodeAccessToken:
    """JWT トークン検証のテスト。"""

    def test_decode_valid_token(self) -> None:
        user_id = uuid.uuid4()
        settings = Settings()
        token = create_access_token(user_id, "u@e.com", "user", settings)
        payload = decode_access_token(token, settings)
        assert payload.sub == str(user_id)
        assert payload.email == "u@e.com"
        assert payload.role == "user"

    def test_expired_token_raises(self) -> None:
        from jose import JWTError

        settings = Settings()
        expired_claims = {
            "sub": str(uuid.uuid4()),
            "email": "u@e.com",
            "role": "user",
            "exp": datetime.now(UTC) - timedelta(hours=1),
        }
        token = jwt.encode(
            expired_claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        with pytest.raises(JWTError):
            decode_access_token(token, settings)

    def test_wrong_secret_raises(self) -> None:
        from jose import JWTError

        settings = Settings()
        token = create_access_token(uuid.uuid4(), "u@e.com", "user", settings)
        bad_settings = Settings(jwt_secret_key="wrong-secret")
        with pytest.raises(JWTError):
            decode_access_token(token, bad_settings)

    def test_malformed_token_raises(self) -> None:
        from jose import JWTError

        settings = Settings()
        with pytest.raises(JWTError):
            decode_access_token("not-a-jwt-token", settings)


# --- get_current_user (FastAPI DI) ---


def _make_auth_app() -> FastAPI:
    """テスト用 FastAPI アプリを生成する。"""
    app = FastAPI()

    @app.get("/me")
    async def me(
        user: Annotated[TokenPayload, Depends(get_current_user)],
    ) -> dict[str, str]:
        return {"sub": user.sub, "email": user.email, "role": user.role}

    return app


class TestGetCurrentUser:
    """get_current_user 依存関数のテスト。"""

    def test_valid_token_returns_user(self) -> None:
        user_id = uuid.uuid4()
        token = create_access_token(user_id, "u@e.com", "admin")
        app = _make_auth_app()
        client = TestClient(app)
        resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sub"] == str(user_id)
        assert data["email"] == "u@e.com"
        assert data["role"] == "admin"

    def test_invalid_token_returns_401(self) -> None:
        app = _make_auth_app()
        client = TestClient(app)
        resp = client.get("/me", headers={"Authorization": "Bearer invalid-token"})
        assert resp.status_code == 401

    def test_expired_token_returns_401(self) -> None:
        settings = Settings()
        expired_claims = {
            "sub": str(uuid.uuid4()),
            "email": "u@e.com",
            "role": "user",
            "exp": datetime.now(UTC) - timedelta(hours=1),
        }
        token = jwt.encode(
            expired_claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        app = _make_auth_app()
        client = TestClient(app)
        resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_no_auth_header_returns_401(self) -> None:
        """HTTPBearer のデフォルト動作: ヘッダーなしは 401。"""
        app = _make_auth_app()
        client = TestClient(app)
        resp = client.get("/me")
        assert resp.status_code == 401

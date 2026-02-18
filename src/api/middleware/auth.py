"""JWT認証・パスワードハッシュモジュール。

bcrypt によるパスワードハッシュ、JWT トークン生成・検証、
get_current_user FastAPI 依存関数を提供する。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from src.config import Settings

_bearer_scheme = HTTPBearer()


def hash_password(plain_password: str) -> str:
    """平文パスワードを bcrypt でハッシュ化する。

    Args:
        plain_password: 平文パスワード。

    Returns:
        bcrypt ハッシュ文字列。
    """
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """平文パスワードをハッシュと照合する。

    Args:
        plain_password: 検証する平文パスワード。
        hashed_password: 保存済みハッシュ。

    Returns:
        一致すれば True。
    """
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


class TokenPayload(BaseModel):
    """JWT トークンのペイロード。"""

    sub: str
    email: str
    role: str
    exp: datetime


def _get_settings() -> Settings:
    """Settings インスタンスを返す内部ヘルパー。"""
    return Settings()


def create_access_token(
    user_id: UUID,
    email: str,
    role: str,
    settings: Settings | None = None,
) -> str:
    """JWT アクセストークンを生成する。

    Args:
        user_id: ユーザー ID。
        email: メールアドレス。
        role: ユーザーロール。
        settings: アプリケーション設定 (省略時はデフォルト)。

    Returns:
        エンコード済み JWT 文字列。
    """
    if settings is None:
        settings = Settings()
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    claims = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": expire,
    }
    encoded: str = jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded


def decode_access_token(token: str, settings: Settings | None = None) -> TokenPayload:
    """JWT トークンをデコード・検証する。

    Args:
        token: エンコード済み JWT 文字列。
        settings: アプリケーション設定 (省略時はデフォルト)。

    Returns:
        検証済み TokenPayload。

    Raises:
        JWTError: トークンが無効・期限切れの場合。
    """
    if settings is None:
        settings = Settings()
    decoded = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    return TokenPayload(**decoded)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    settings: Annotated[Settings, Depends(_get_settings)],
) -> TokenPayload:
    """Bearer トークンから現在のユーザーを取得する FastAPI 依存関数。

    Args:
        credentials: HTTPBearer から取得した認証情報。
        settings: アプリケーション設定。

    Returns:
        検証済み TokenPayload。

    Raises:
        HTTPException: トークンが無効・期限切れの場合 (401)。
    """
    try:
        return decode_access_token(credentials.credentials, settings)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

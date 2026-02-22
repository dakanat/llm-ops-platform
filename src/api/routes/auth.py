"""ユーザー登録 API エンドポイント。"""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.middleware.auth import create_access_token, hash_password
from src.db.models import User
from src.db.session import get_session
from src.security.audit_log import log_action

router = APIRouter(prefix="/auth")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    """ユーザー登録リクエスト。"""

    email: str = Field(min_length=1)
    name: str = Field(min_length=1)
    password: str = Field(min_length=8)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """メール形式の検証と小文字正規化。"""
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            msg = "Invalid email format"
            raise ValueError(msg)
        return v


class RegisterResponse(BaseModel):
    """ユーザー登録レスポンス。"""

    id: uuid.UUID
    email: str
    name: str
    role: str
    access_token: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    request: RegisterRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RegisterResponse:
    """新規ユーザーを登録する。認証不要の公開エンドポイント。"""
    user = User(
        email=request.email,
        name=request.name,
        hashed_password=hash_password(request.password),
        role="user",
    )
    session.add(user)

    await log_action(
        session=session,
        user_id=user.id,
        action="register",
        resource_type="user",
        resource_id=str(user.id),
    )

    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{request.email}' is already registered",
        ) from e

    token = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
    )

    return RegisterResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        access_token=token,
    )

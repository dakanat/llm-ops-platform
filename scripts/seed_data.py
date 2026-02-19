"""初期ユーザー投入スクリプト。

admin / user / viewer の3ユーザーをDBに投入する。
既存ユーザーはスキップし、冪等に実行できる。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession
from src.api.middleware.auth import hash_password
from src.config import Settings
from src.db.models import User
from src.monitoring.logger import get_logger, setup_logging


class SeedUserData(BaseModel):
    """シードユーザーの定義。

    Attributes:
        email: メールアドレス。
        name: ユーザー名。
        role: ロール (admin / user / viewer)。
        password: 平文パスワード。
    """

    email: str
    name: str
    role: str
    password: str


SEED_USERS: list[SeedUserData] = [
    SeedUserData(
        email="admin@example.com",
        name="Admin User",
        role="admin",
        password="admin-password",
    ),
    SeedUserData(
        email="user@example.com",
        name="Regular User",
        role="user",
        password="user-password",
    ),
    SeedUserData(
        email="viewer@example.com",
        name="Viewer User",
        role="viewer",
        password="viewer-password",
    ),
]


async def seed_users(session: SQLModelAsyncSession) -> list[str]:
    """シードユーザーをDBに投入する。

    各ユーザーの存在を確認し、未存在なら作成、既存ならスキップする。

    Args:
        session: 非同期DBセッション。

    Returns:
        各ユーザーの処理結果リスト (例: ["created:admin@example.com", "skipped:user@example.com"])。
    """
    logger = get_logger(script="seed_data")
    results: list[str] = []

    for seed_user in SEED_USERS:
        statement = select(User).where(User.email == seed_user.email)
        result = await session.exec(statement)
        existing = result.first()

        if existing is not None:
            logger.info("user_skipped", email=seed_user.email)
            results.append(f"skipped:{seed_user.email}")
        else:
            hashed = hash_password(seed_user.password)
            user = User(
                email=seed_user.email,
                name=seed_user.name,
                role=seed_user.role,
                hashed_password=hashed,
            )
            session.add(user)
            logger.info("user_created", email=seed_user.email, role=seed_user.role)
            results.append(f"created:{seed_user.email}")

    await session.commit()
    return results


async def main() -> None:
    """エントリポイント。DBに接続してシードユーザーを投入する。"""
    settings = Settings()
    setup_logging(settings.log_level)
    logger = get_logger(script="seed_data")

    logger.info("seed_start")
    engine = create_async_engine(settings.database_url)

    async with SQLModelAsyncSession(engine) as session:
        results = await seed_users(session)

    for result in results:
        logger.info("seed_result", result=result)

    logger.info("seed_complete")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

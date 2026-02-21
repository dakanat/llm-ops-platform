"""会話の永続化サービス。"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.models import Conversation, Message


class ConversationService:
    """会話とメッセージの CRUD 操作を提供する。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_conversation(
        self,
        user_id: uuid.UUID,
        title: str | None = None,
    ) -> Conversation:
        """新規会話を作成する。

        Args:
            user_id: ユーザー ID。
            title: 会話タイトル（省略可）。

        Returns:
            作成された Conversation。
        """
        conv = Conversation(user_id=user_id, title=title)
        self._session.add(conv)
        await self._session.commit()
        await self._session.refresh(conv)
        return conv

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
    ) -> list[Conversation]:
        """ユーザーの会話一覧を取得する（新しい順）。

        Args:
            user_id: ユーザー ID。
            limit: 最大取得件数。

        Returns:
            Conversation のリスト。
        """
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)  # type: ignore[arg-type]
            .order_by(Conversation.updated_at.desc())  # type: ignore[attr-defined]
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Conversation | None:
        """会話を取得する（ユーザー所有権チェック付き）。

        Args:
            conversation_id: 会話 ID。
            user_id: ユーザー ID。

        Returns:
            Conversation または None。
        """
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,  # type: ignore[arg-type]
            Conversation.user_id == user_id,  # type: ignore[arg-type]
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
    ) -> Message:
        """会話にメッセージを追加する。

        Args:
            conversation_id: 会話 ID。
            role: メッセージの役割（user / assistant）。
            content: メッセージ内容。

        Returns:
            作成された Message。
        """
        msg = Message(conversation_id=conversation_id, role=role, content=content)
        self._session.add(msg)
        await self._session.commit()
        return msg

    async def get_messages(
        self,
        conversation_id: uuid.UUID,
    ) -> list[Message]:
        """会話のメッセージ一覧を取得する（時系列順）。

        Args:
            conversation_id: 会話 ID。

        Returns:
            Message のリスト。
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)  # type: ignore[arg-type]
            .order_by(Message.created_at.asc())  # type: ignore[attr-defined]
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_title(
        self,
        conversation_id: uuid.UUID,
        title: str,
    ) -> None:
        """会話タイトルを更新する。

        Args:
            conversation_id: 会話 ID。
            title: 新しいタイトル。
        """
        conv = await self._session.get(Conversation, conversation_id)
        if conv is not None:
            conv.title = title
            await self._session.commit()

    async def delete_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """会話を削除する。

        Args:
            conversation_id: 会話 ID。
            user_id: ユーザー ID（所有権チェック）。

        Returns:
            削除成功なら True。
        """
        conv = await self.get_conversation(conversation_id, user_id)
        if conv is None:
            return False
        await self._session.delete(conv)
        await self._session.commit()
        return True

"""Tests for ConversationService."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.db.models import Conversation, Message
from src.web.services.conversation import ConversationService


@pytest.fixture
def user_id() -> uuid.UUID:
    """Return a test user ID."""
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def conversation_id() -> uuid.UUID:
    """Return a test conversation ID."""
    return uuid.UUID("00000000-0000-0000-0000-000000000010")


@pytest.fixture
def mock_session() -> AsyncMock:
    """Return a mock async DB session."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def service(mock_session: AsyncMock) -> ConversationService:
    """Return a ConversationService with a mock session."""
    return ConversationService(session=mock_session)


class TestCreateConversation:
    """ConversationService.create_conversation のテスト。"""

    async def test_creates_conversation_with_user_id(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        user_id: uuid.UUID,
    ) -> None:
        result = await service.create_conversation(user_id)
        assert isinstance(result, Conversation)
        assert result.user_id == user_id
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    async def test_creates_conversation_with_title(
        self,
        service: ConversationService,
        user_id: uuid.UUID,
    ) -> None:
        result = await service.create_conversation(user_id, title="Test Chat")
        assert result.title == "Test Chat"

    async def test_creates_conversation_without_title(
        self,
        service: ConversationService,
        user_id: uuid.UUID,
    ) -> None:
        result = await service.create_conversation(user_id)
        assert result.title is None


class TestListConversations:
    """ConversationService.list_conversations のテスト。"""

    async def test_returns_conversations_list(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        user_id: uuid.UUID,
    ) -> None:
        conv = Conversation(user_id=user_id, title="Chat 1")
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [conv]
        mock_session.execute.return_value = mock_result

        result = await service.list_conversations(user_id)
        assert len(result) == 1
        assert result[0].title == "Chat 1"

    async def test_respects_limit_parameter(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        user_id: uuid.UUID,
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        await service.list_conversations(user_id, limit=10)
        mock_session.execute.assert_awaited_once()


class TestGetConversation:
    """ConversationService.get_conversation のテスト。"""

    async def test_returns_conversation_when_found(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> None:
        conv = Conversation(id=conversation_id, user_id=user_id, title="Found")
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = conv
        mock_session.execute.return_value = mock_result

        result = await service.get_conversation(conversation_id, user_id)
        assert result is not None
        assert result.title == "Found"

    async def test_returns_none_when_not_found(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute.return_value = mock_result

        result = await service.get_conversation(conversation_id, user_id)
        assert result is None


class TestAddMessage:
    """ConversationService.add_message のテスト。"""

    async def test_adds_message_with_correct_fields(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        conversation_id: uuid.UUID,
    ) -> None:
        result = await service.add_message(conversation_id, "user", "Hello")
        assert isinstance(result, Message)
        assert result.conversation_id == conversation_id
        assert result.role == "user"
        assert result.content == "Hello"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    async def test_adds_assistant_message(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        conversation_id: uuid.UUID,
    ) -> None:
        result = await service.add_message(conversation_id, "assistant", "Hi there")
        assert result.role == "assistant"
        assert result.content == "Hi there"


class TestGetMessages:
    """ConversationService.get_messages のテスト。"""

    async def test_returns_messages_list(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        conversation_id: uuid.UUID,
    ) -> None:
        msg1 = Message(conversation_id=conversation_id, role="user", content="Hi")
        msg2 = Message(conversation_id=conversation_id, role="assistant", content="Hello")
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg1, msg2]
        mock_session.execute.return_value = mock_result

        result = await service.get_messages(conversation_id)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"


class TestUpdateTitle:
    """ConversationService.update_title のテスト。"""

    async def test_updates_title(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        conversation_id: uuid.UUID,
    ) -> None:
        conv = Conversation(
            id=conversation_id,
            user_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        )
        mock_session.get.return_value = conv

        await service.update_title(conversation_id, "New Title")
        assert conv.title == "New Title"
        mock_session.commit.assert_awaited_once()


class TestDeleteConversation:
    """ConversationService.delete_conversation のテスト。"""

    async def test_deletes_existing_conversation(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> None:
        conv = Conversation(id=conversation_id, user_id=user_id)
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = conv
        mock_session.execute.return_value = mock_result

        result = await service.delete_conversation(conversation_id, user_id)
        assert result is True
        mock_session.delete.assert_awaited_once_with(conv)

    async def test_returns_false_when_not_found(
        self,
        service: ConversationService,
        mock_session: AsyncMock,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> None:
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute.return_value = mock_result

        result = await service.delete_conversation(conversation_id, user_id)
        assert result is False

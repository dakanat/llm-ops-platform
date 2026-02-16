"""Tests for SQLModel database models."""

import uuid
from datetime import UTC, datetime

import pytest


class TestUserModel:
    """User モデルのインスタンス化・バリデーションテスト。"""

    def test_user_creates_with_required_fields(self) -> None:
        """必須フィールドで User が生成できること。"""
        from src.db.models import User

        user = User(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed",
        )

        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.hashed_password == "hashed"

    def test_user_id_defaults_to_uuid(self) -> None:
        """id がデフォルトで UUID を生成すること。"""
        from src.db.models import User

        user = User(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed",
        )

        assert user.id is not None
        assert isinstance(user.id, uuid.UUID)

    def test_user_role_accepts_valid_values(self) -> None:
        """role が admin/user/viewer を受け付けること。"""
        from src.db.models import User

        for role in ("admin", "user", "viewer"):
            user = User(
                email="test@example.com",
                name="Test User",
                hashed_password="hashed",
                role=role,
            )
            assert user.role == role

    def test_user_role_rejects_invalid_value(self) -> None:
        """不正な role が拒否されること。"""
        from src.db.models import User

        with pytest.raises(ValueError, match="role"):
            User(
                email="test@example.com",
                name="Test User",
                hashed_password="hashed",
                role="superadmin",
            )

    def test_user_role_defaults_to_user(self) -> None:
        """role のデフォルト値が 'user' であること。"""
        from src.db.models import User

        user = User(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed",
        )

        assert user.role == "user"

    def test_user_created_at_defaults(self) -> None:
        """created_at がデフォルトで現在時刻付近を返すこと。"""
        from src.db.models import User

        before = datetime.now(UTC)
        user = User(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed",
        )
        after = datetime.now(UTC)

        assert user.created_at is not None
        assert before <= user.created_at <= after

    def test_user_is_table_model(self) -> None:
        """User が SQLModel テーブルモデルであること。"""
        from src.db.models import User

        assert hasattr(User, "__tablename__")
        assert User.__tablename__ == "users"

    def test_user_is_active_defaults_to_true(self) -> None:
        """is_active のデフォルト値が True であること。"""
        from src.db.models import User

        user = User(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed",
        )

        assert user.is_active is True


class TestDocumentModel:
    """Document モデルのインスタンス化・バリデーションテスト。"""

    def test_document_creates_with_required_fields(self) -> None:
        """必須フィールドで Document が生成できること。"""
        from src.db.models import Document

        user_id = uuid.uuid4()
        doc = Document(
            title="Test Doc",
            content="Test content",
            user_id=user_id,
        )

        assert doc.title == "Test Doc"
        assert doc.content == "Test content"
        assert doc.user_id == user_id

    def test_document_metadata_defaults_to_empty_dict(self) -> None:
        """metadata_ のデフォルト値が空の dict であること。"""
        from src.db.models import Document

        doc = Document(
            title="Test Doc",
            content="Test content",
            user_id=uuid.uuid4(),
        )

        assert doc.metadata_ == {}

    def test_document_metadata_can_be_set(self) -> None:
        """metadata_ に値を設定できること。"""
        from src.db.models import Document

        metadata = {"source": "web", "language": "ja"}
        doc = Document(
            title="Test Doc",
            content="Test content",
            user_id=uuid.uuid4(),
            metadata_=metadata,
        )

        assert doc.metadata_ == metadata

    def test_document_is_table_model(self) -> None:
        """Document が SQLModel テーブルモデルであること。"""
        from src.db.models import Document

        assert hasattr(Document, "__tablename__")
        assert Document.__tablename__ == "documents"


class TestChunkModel:
    """Chunk モデルのインスタンス化・バリデーションテスト。"""

    def test_chunk_creates_with_required_fields(self) -> None:
        """必須フィールドで Chunk が生成できること。"""
        from src.db.models import Chunk

        document_id = uuid.uuid4()
        chunk = Chunk(
            document_id=document_id,
            content="Chunk content",
            chunk_index=0,
        )

        assert chunk.document_id == document_id
        assert chunk.content == "Chunk content"
        assert chunk.chunk_index == 0

    def test_chunk_embedding_defaults_to_none(self) -> None:
        """embedding のデフォルト値が None であること。"""
        from src.db.models import Chunk

        chunk = Chunk(
            document_id=uuid.uuid4(),
            content="Chunk content",
            chunk_index=0,
        )

        assert chunk.embedding is None

    def test_chunk_embedding_can_be_set(self) -> None:
        """embedding に list[float] を設定できること。"""
        from src.db.models import Chunk

        embedding = [0.1] * 1024
        chunk = Chunk(
            document_id=uuid.uuid4(),
            content="Chunk content",
            chunk_index=0,
            embedding=embedding,
        )

        assert chunk.embedding == embedding

    def test_chunk_is_table_model(self) -> None:
        """Chunk が SQLModel テーブルモデルであること。"""
        from src.db.models import Chunk

        assert hasattr(Chunk, "__tablename__")
        assert Chunk.__tablename__ == "chunks"


class TestConversationModel:
    """Conversation モデルのインスタンス化・バリデーションテスト。"""

    def test_conversation_creates_with_required_fields(self) -> None:
        """必須フィールドで Conversation が生成できること。"""
        from src.db.models import Conversation

        user_id = uuid.uuid4()
        conv = Conversation(user_id=user_id)

        assert conv.user_id == user_id

    def test_conversation_title_is_optional(self) -> None:
        """title がオプショナルで None がデフォルトであること。"""
        from src.db.models import Conversation

        conv = Conversation(user_id=uuid.uuid4())

        assert conv.title is None

    def test_conversation_title_can_be_set(self) -> None:
        """title に値を設定できること。"""
        from src.db.models import Conversation

        conv = Conversation(user_id=uuid.uuid4(), title="Test Conversation")

        assert conv.title == "Test Conversation"

    def test_conversation_is_table_model(self) -> None:
        """Conversation が SQLModel テーブルモデルであること。"""
        from src.db.models import Conversation

        assert hasattr(Conversation, "__tablename__")
        assert Conversation.__tablename__ == "conversations"


class TestMessageModel:
    """Message モデルのインスタンス化・バリデーションテスト。"""

    def test_message_creates_with_required_fields(self) -> None:
        """必須フィールドで Message が生成できること。"""
        from src.db.models import Message

        conversation_id = uuid.uuid4()
        msg = Message(
            conversation_id=conversation_id,
            role="user",
            content="Hello",
        )

        assert msg.conversation_id == conversation_id
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_role_accepts_valid_values(self) -> None:
        """role が system/user/assistant を受け付けること。"""
        from src.db.models import Message

        for role in ("system", "user", "assistant"):
            msg = Message(
                conversation_id=uuid.uuid4(),
                role=role,
                content="Hello",
            )
            assert msg.role == role

    def test_message_role_rejects_invalid_value(self) -> None:
        """不正な role が拒否されること。"""
        from src.db.models import Message

        with pytest.raises(ValueError, match="role"):
            Message(
                conversation_id=uuid.uuid4(),
                role="bot",
                content="Hello",
            )

    def test_message_token_count_defaults_to_none(self) -> None:
        """token_count のデフォルト値が None であること。"""
        from src.db.models import Message

        msg = Message(
            conversation_id=uuid.uuid4(),
            role="user",
            content="Hello",
        )

        assert msg.token_count is None

    def test_message_is_table_model(self) -> None:
        """Message が SQLModel テーブルモデルであること。"""
        from src.db.models import Message

        assert hasattr(Message, "__tablename__")
        assert Message.__tablename__ == "messages"


class TestAuditLogModel:
    """AuditLog モデルのインスタンス化・バリデーションテスト。"""

    def test_audit_log_creates_with_required_fields(self) -> None:
        """必須フィールドで AuditLog が生成できること。"""
        from src.db.models import AuditLog

        user_id = uuid.uuid4()
        log = AuditLog(
            user_id=user_id,
            action="create",
            resource_type="document",
            resource_id="doc-123",
        )

        assert log.user_id == user_id
        assert log.action == "create"
        assert log.resource_type == "document"
        assert log.resource_id == "doc-123"

    def test_audit_log_details_defaults_to_empty_dict(self) -> None:
        """details のデフォルト値が空の dict であること。"""
        from src.db.models import AuditLog

        log = AuditLog(
            user_id=uuid.uuid4(),
            action="create",
            resource_type="document",
            resource_id="doc-123",
        )

        assert log.details == {}

    def test_audit_log_details_can_be_set(self) -> None:
        """details に値を設定できること。"""
        from src.db.models import AuditLog

        details = {"ip": "127.0.0.1", "changes": ["title"]}
        log = AuditLog(
            user_id=uuid.uuid4(),
            action="update",
            resource_type="document",
            resource_id="doc-123",
            details=details,
        )

        assert log.details == details

    def test_audit_log_is_table_model(self) -> None:
        """AuditLog が SQLModel テーブルモデルであること。"""
        from src.db.models import AuditLog

        assert hasattr(AuditLog, "__tablename__")
        assert AuditLog.__tablename__ == "audit_logs"

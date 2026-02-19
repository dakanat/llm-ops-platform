"""SQLModel database models for the LLM Ops Platform."""

import uuid
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """User account model."""

    __tablename__ = "users"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str
    hashed_password: str
    role: str = Field(default="user")
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: object) -> None:
        """Validate role after initialization."""
        allowed = ("admin", "user", "viewer")
        if self.role not in allowed:
            msg = f"role must be one of {allowed}, got '{self.role}'"
            raise ValueError(msg)


class Document(SQLModel, table=True):
    """Document model for RAG pipeline."""

    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    title: str
    content: str = Field(sa_column=Column(Text, nullable=False))
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False, default={}),
    )
    user_id: uuid.UUID = Field(foreign_key="users.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(SQLModel, table=True):
    """Chunk model for document chunks with vector embeddings."""

    __tablename__ = "chunks"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(foreign_key="documents.id")
    content: str = Field(sa_column=Column(Text, nullable=False))
    chunk_index: int
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(1024)),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Conversation(SQLModel, table=True):
    """Conversation model for chat sessions."""

    __tablename__ = "conversations"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="users.id")
    title: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Message(SQLModel, table=True):
    """Message model for conversation messages."""

    __tablename__ = "messages"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    conversation_id: uuid.UUID = Field(foreign_key="conversations.id")
    role: str
    content: str = Field(sa_column=Column(Text, nullable=False))
    token_count: int | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: object) -> None:
        """Validate role after initialization."""
        allowed = ("system", "user", "assistant")
        if self.role not in allowed:
            msg = f"role must be one of {allowed}, got '{self.role}'"
            raise ValueError(msg)


class AuditLog(SQLModel, table=True):
    """Audit log model for tracking user actions."""

    __tablename__ = "audit_logs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="users.id")
    action: str
    resource_type: str
    resource_id: str
    details: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, default={}),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvalDatasetRecord(SQLModel, table=True):
    """Evaluation dataset record."""

    __tablename__ = "eval_datasets"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str | None = Field(default=None, sa_column=Column(Text))
    created_by: uuid.UUID = Field(foreign_key="users.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvalExampleRecord(SQLModel, table=True):
    """Evaluation example record belonging to a dataset."""

    __tablename__ = "eval_examples"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    dataset_id: uuid.UUID = Field(foreign_key="eval_datasets.id")
    query: str = Field(sa_column=Column(Text, nullable=False))
    context: str = Field(sa_column=Column(Text, nullable=False))
    answer: str = Field(sa_column=Column(Text, nullable=False))
    expected_answer: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

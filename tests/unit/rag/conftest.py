"""Shared fixtures and helpers for RAG tests."""

from __future__ import annotations

import uuid

from src.db.models import Chunk, Document
from src.llm.providers.base import LLMResponse, TokenUsage
from src.rag.chunker import TextChunk
from src.rag.retriever import RetrievedChunk

EMBEDDING_DIM = 1024


def make_document(
    content: str = "テスト文書の内容です。",
    title: str = "テスト文書",
) -> Document:
    """テスト用の Document インスタンスを生成。"""
    return Document(
        id=uuid.uuid4(),
        title=title,
        content=content,
        user_id=uuid.uuid4(),
    )


def make_chunk(
    document_id: uuid.UUID | None = None,
    chunk_index: int = 0,
    content: str = "テストチャンク",
) -> Chunk:
    """テスト用の Chunk インスタンスを生成。"""
    return Chunk(
        id=uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        content=content,
        chunk_index=chunk_index,
        embedding=[0.1] * EMBEDDING_DIM,
    )


def make_retrieved_chunk(
    content: str = "テストチャンク",
    chunk_index: int = 0,
    document_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    """テスト用の RetrievedChunk インスタンスを生成。"""
    return RetrievedChunk(
        content=content,
        chunk_index=chunk_index,
        document_id=document_id or uuid.uuid4(),
    )


def make_text_chunk(
    content: str = "テストチャンク",
    index: int = 0,
    start: int = 0,
    end: int | None = None,
) -> TextChunk:
    """テスト用の TextChunk インスタンスを生成。"""
    return TextChunk(
        content=content,
        index=index,
        start=start,
        end=end if end is not None else start + len(content),
    )


def make_llm_response(
    content: str = "回答テキスト",
    model: str = "test-model",
    usage: TokenUsage | None = None,
) -> LLMResponse:
    """テスト用の LLMResponse を生成。"""
    return LLMResponse(
        content=content,
        model=model,
        usage=usage,
    )

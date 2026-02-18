"""Tests for PIISanitizingProvider — LLMプロバイダラッパー。

PIIを含むメッセージがマスクされてから内部プロバイダに渡されること、
レスポンスはそのまま返されること、無効時はパススルーすることを検証する。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from src.llm.providers.base import ChatMessage, LLMChunk, LLMResponse, Role, TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inner_provider(
    complete_response: LLMResponse | None = None,
    stream_chunks: list[LLMChunk] | None = None,
) -> Any:
    """Create a mock inner LLM provider."""
    provider = AsyncMock()

    if complete_response is None:
        complete_response = LLMResponse(
            content="response text",
            model="test-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    provider.complete.return_value = complete_response

    if stream_chunks is None:
        stream_chunks = [
            LLMChunk(content="chunk1"),
            LLMChunk(content="chunk2", finish_reason="stop"),
        ]

    async def mock_stream(
        messages: list[ChatMessage], model: str, **kwargs: Any
    ) -> AsyncGenerator[LLMChunk, None]:
        for chunk in stream_chunks:
            yield chunk

    provider.stream = mock_stream

    return provider


# ---------------------------------------------------------------------------
# complete() — PII masking on outbound messages
# ---------------------------------------------------------------------------


class TestCompleteMasksPII:
    """complete() がユーザーメッセージ中のPIIをマスクして内部プロバイダに渡すこと。"""

    @pytest.mark.asyncio
    async def test_complete_masks_pii_in_user_message_content(self) -> None:
        """ユーザーメッセージ内のメールアドレスがマスクされること。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        inner = _make_inner_provider()
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [ChatMessage(role=Role.user, content="連絡先: user@example.com")]
        await wrapper.complete(messages, model="test-model")

        called_messages = inner.complete.call_args[0][0]
        assert "user@example.com" not in called_messages[0].content
        assert "[EMAIL]" in called_messages[0].content

    @pytest.mark.asyncio
    async def test_complete_preserves_system_message_without_pii(self) -> None:
        """PIIを含まないシステムメッセージがそのまま渡されること。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        inner = _make_inner_provider()
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [
            ChatMessage(role=Role.system, content="You are a helpful assistant."),
            ChatMessage(role=Role.user, content="Hello"),
        ]
        await wrapper.complete(messages, model="test-model")

        called_messages = inner.complete.call_args[0][0]
        assert called_messages[0].content == "You are a helpful assistant."
        assert called_messages[1].content == "Hello"

    @pytest.mark.asyncio
    async def test_complete_passes_clean_messages_unchanged(self) -> None:
        """PIIを含まないメッセージがそのまま渡されること。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        inner = _make_inner_provider()
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [ChatMessage(role=Role.user, content="What is the weather?")]
        await wrapper.complete(messages, model="test-model")

        called_messages = inner.complete.call_args[0][0]
        assert called_messages[0].content == "What is the weather?"

    @pytest.mark.asyncio
    async def test_complete_does_not_mask_llm_response(self) -> None:
        """LLMのレスポンスはマスクされずにそのまま返されること。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        response = LLMResponse(
            content="Please email user@example.com",
            model="test-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        inner = _make_inner_provider(complete_response=response)
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [ChatMessage(role=Role.user, content="Hello")]
        result = await wrapper.complete(messages, model="test-model")

        assert result.content == "Please email user@example.com"

    @pytest.mark.asyncio
    async def test_complete_masks_multiple_pii_types(self) -> None:
        """複数種類のPIIが同時にマスクされること。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        inner = _make_inner_provider()
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [
            ChatMessage(
                role=Role.user,
                content="連絡先: user@example.com 電話: 090-1234-5678",
            ),
        ]
        await wrapper.complete(messages, model="test-model")

        called_messages = inner.complete.call_args[0][0]
        assert "user@example.com" not in called_messages[0].content
        assert "090-1234-5678" not in called_messages[0].content
        assert "[EMAIL]" in called_messages[0].content
        assert "[PHONE]" in called_messages[0].content


# ---------------------------------------------------------------------------
# stream() — PII masking on outbound messages
# ---------------------------------------------------------------------------


class TestStreamMasksPII:
    """stream() がユーザーメッセージ中のPIIをマスクして内部プロバイダに渡すこと。"""

    @pytest.mark.asyncio
    async def test_stream_masks_pii_in_outbound_messages(self) -> None:
        """stream() でもメッセージ中のPIIがマスクされること。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        captured_messages: list[ChatMessage] = []
        chunks = [LLMChunk(content="hi", finish_reason="stop")]

        async def capturing_stream(
            messages: list[ChatMessage], model: str, **kwargs: Any
        ) -> AsyncGenerator[LLMChunk, None]:
            captured_messages.extend(messages)
            for chunk in chunks:
                yield chunk

        inner = _make_inner_provider()
        inner.stream = capturing_stream
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [ChatMessage(role=Role.user, content="メール: user@example.com")]
        async for _ in wrapper.stream(messages, model="test-model"):
            pass

        assert "user@example.com" not in captured_messages[0].content
        assert "[EMAIL]" in captured_messages[0].content

    @pytest.mark.asyncio
    async def test_stream_yields_chunks_unmodified(self) -> None:
        """stream() がチャンクをそのまま返すこと。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        inner = _make_inner_provider()
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [ChatMessage(role=Role.user, content="Hello")]
        collected = [chunk async for chunk in wrapper.stream(messages, model="test-model")]

        assert len(collected) == 2
        assert collected[0].content == "chunk1"
        assert collected[1].content == "chunk2"


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """無効時はメッセージをそのまま内部プロバイダに渡すこと。"""

    @pytest.mark.asyncio
    async def test_disabled_mode_passes_messages_through(self) -> None:
        """enabled=False のときメッセージがマスクされないこと。"""
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        inner = _make_inner_provider()
        wrapper = PIISanitizingProvider(inner=inner, enabled=False)

        messages = [ChatMessage(role=Role.user, content="連絡先: user@example.com")]
        await wrapper.complete(messages, model="test-model")

        called_messages = inner.complete.call_args[0][0]
        assert called_messages[0].content == "連絡先: user@example.com"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    """PIIマスキング時にPII種別のみがログに記録されること。"""

    @pytest.mark.asyncio
    async def test_logs_pii_types_when_masking(self) -> None:
        """PIIマスキングが発生した場合にログイベントが出力されること。"""
        from unittest.mock import patch

        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        inner = _make_inner_provider()
        wrapper = PIISanitizingProvider(inner=inner)

        messages = [ChatMessage(role=Role.user, content="連絡先: user@example.com")]

        with patch("src.llm.pii_sanitizing_provider.logger") as mock_logger:
            await wrapper.complete(messages, model="test-model")
            mock_logger.info.assert_called_once()
            call_kwargs = mock_logger.info.call_args
            assert "pii_masked_for_llm" in call_kwargs[0]
            assert "pii_types" in call_kwargs[1]

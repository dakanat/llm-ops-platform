"""Tests for LLM provider Protocol, data models, and OpenRouter implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.llm.providers.base import (
    ChatMessage,
    LLMChunk,
    LLMProvider,
    LLMResponse,
    Role,
    TokenUsage,
)
from src.llm.providers.openrouter import OpenRouterProvider


class TestChatMessage:
    """ChatMessage が正しく生成・シリアライズできること。"""

    def test_creates_with_valid_role(self) -> None:
        """有効な role で ChatMessage が生成できること。"""
        msg = ChatMessage(role=Role.user, content="Hello")

        assert msg.role == Role.user
        assert msg.content == "Hello"

    def test_model_dump_serializes_role_as_string(self) -> None:
        """model_dump() で role が文字列にシリアライズされること。"""
        msg = ChatMessage(role=Role.assistant, content="Hi")
        dumped = msg.model_dump()

        assert dumped["role"] == "assistant"
        assert isinstance(dumped["role"], str)

    def test_creates_with_each_role(self) -> None:
        """system, user, assistant の各 role で生成できること。"""
        for role in Role:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role


class TestLLMResponse:
    """LLMResponse が必須・オプションフィールドで生成できること。"""

    def test_creates_with_required_fields_only(self) -> None:
        """content と model のみで生成できること。"""
        resp = LLMResponse(content="Hello", model="openai/gpt-oss-120b:free")

        assert resp.content == "Hello"
        assert resp.model == "openai/gpt-oss-120b:free"
        assert resp.usage is None
        assert resp.finish_reason is None

    def test_creates_with_token_usage(self) -> None:
        """TokenUsage 付きで生成できること。"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        resp = LLMResponse(content="Hi", model="test-model", usage=usage, finish_reason="stop")

        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 20
        assert resp.usage.total_tokens == 30
        assert resp.finish_reason == "stop"


class TestLLMChunk:
    """LLMChunk が正しく生成できること。"""

    def test_creates_with_content(self) -> None:
        """content で生成できること。"""
        chunk = LLMChunk(content="Hello")

        assert chunk.content == "Hello"
        assert chunk.finish_reason is None

    def test_creates_with_finish_reason(self) -> None:
        """finish_reason 付きで生成できること。"""
        chunk = LLMChunk(content="", finish_reason="stop")

        assert chunk.finish_reason == "stop"


class TestLLMProviderProtocol:
    """LLMProvider Protocol の型チェック。"""

    def test_openrouter_provider_satisfies_protocol(self) -> None:
        """OpenRouterProvider が LLMProvider Protocol を満たすこと。"""
        provider = OpenRouterProvider(api_key="test-key")

        assert isinstance(provider, LLMProvider)

    def test_incomplete_class_does_not_satisfy_protocol(self) -> None:
        """complete/stream を持たないクラスは Protocol を満たさないこと。"""

        class Incomplete:
            pass

        assert not isinstance(Incomplete(), LLMProvider)


class TestOpenRouterProviderComplete:
    """OpenRouterProvider.complete() のテスト。"""

    @pytest.fixture
    def provider(self) -> OpenRouterProvider:
        return OpenRouterProvider(api_key="test-api-key")

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        """httpx.Response のモックを作成。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, provider: OpenRouterProvider) -> None:
        """正常なレスポンスから LLMResponse が返ること。"""
        response_data = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "model": "openai/gpt-oss-120b:free",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [ChatMessage(role=Role.user, content="Hi")]
            result = await provider.complete(messages, model="openai/gpt-oss-120b:free")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.model == "openai/gpt-oss-120b:free"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_sends_correct_payload(self, provider: OpenRouterProvider) -> None:
        """正しいペイロードが送信されること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "openai/gpt-oss-120b:free",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [
                ChatMessage(role=Role.system, content="You are helpful."),
                ChatMessage(role=Role.user, content="Hi"),
            ]
            await provider.complete(messages, model="openai/gpt-oss-120b:free")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "openai/gpt-oss-120b:free"
        assert payload["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        assert payload.get("stream") is not True

    @pytest.mark.asyncio
    async def test_complete_sends_authorization_header(self) -> None:
        """Authorization ヘッダーが正しく設定されること。"""
        provider = OpenRouterProvider(api_key="sk-test-123")

        assert provider._client.headers["authorization"] == "Bearer sk-test-123"

    @pytest.mark.asyncio
    async def test_complete_parses_usage(self, provider: OpenRouterProvider) -> None:
        """usage が正しくパースされること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "test",
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")], model="test"
            )

        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 300

    @pytest.mark.asyncio
    async def test_complete_raises_on_http_error(self, provider: OpenRouterProvider) -> None:
        """HTTPエラー時に例外が発生すること。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(spec=httpx.Request),
            response=MagicMock(spec=httpx.Response),
        )

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            with pytest.raises(httpx.HTTPStatusError):
                await provider.complete([ChatMessage(role=Role.user, content="Hi")], model="test")

    @pytest.mark.asyncio
    async def test_complete_forwards_kwargs(self, provider: OpenRouterProvider) -> None:
        """kwargs が正しくペイロードに伝播されること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "test",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")],
                model="test",
                temperature=0.5,
                max_tokens=100,
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100


class TestOpenRouterProviderStream:
    """OpenRouterProvider.stream() のテスト。"""

    @pytest.fixture
    def provider(self) -> OpenRouterProvider:
        return OpenRouterProvider(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider: OpenRouterProvider) -> None:
        """ストリーミングで LLMChunk が yield されること。"""
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider._client, "stream", return_value=mock_stream_cm):
            chunks = []
            async for chunk in provider.stream(
                [ChatMessage(role=Role.user, content="Hi")], model="test"
            ):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self, provider: OpenRouterProvider) -> None:
        """data: [DONE] でストリーミングが終了すること。"""
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            b"data: [DONE]",
            b'data: {"choices":[{"delta":{"content":"should not appear"},"finish_reason":null}]}',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider._client, "stream", return_value=mock_stream_cm):
            chunks = []
            async for chunk in provider.stream(
                [ChatMessage(role=Role.user, content="Hi")], model="test"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hi"

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self, provider: OpenRouterProvider) -> None:
        """空の content を持つ delta はスキップされること。"""
        lines = [
            b'data: {"choices":[{"delta":{"content":""},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":null}]}',
            b"data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider._client, "stream", return_value=mock_stream_cm):
            chunks = []
            async for chunk in provider.stream(
                [ChatMessage(role=Role.user, content="Hi")], model="test"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"

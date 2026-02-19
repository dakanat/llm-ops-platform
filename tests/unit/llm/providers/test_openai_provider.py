"""Tests for OpenAI LLM provider implementation."""

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
)
from src.llm.providers.openai_provider import OpenAIProvider


class TestOpenAIProviderProtocol:
    """OpenAIProvider が LLMProvider Protocol を満たすこと。"""

    def test_openai_provider_satisfies_protocol(self) -> None:
        """OpenAIProvider が LLMProvider Protocol を満たすこと。"""
        provider = OpenAIProvider(api_key="test-key")

        assert isinstance(provider, LLMProvider)


class TestOpenAIProviderComplete:
    """OpenAIProvider.complete() のテスト。"""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="test-api-key")

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        """httpx.Response のモックを作成。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, provider: OpenAIProvider) -> None:
        """正常なレスポンスから LLMResponse が返ること。"""
        response_data = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [ChatMessage(role=Role.user, content="Hi")]
            result = await provider.complete(messages, model="gpt-4o")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.model == "gpt-4o"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_sends_correct_payload(self, provider: OpenAIProvider) -> None:
        """正しいペイロードが送信されること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [
                ChatMessage(role=Role.system, content="You are helpful."),
                ChatMessage(role=Role.user, content="Hi"),
            ]
            await provider.complete(messages, model="gpt-4o")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        assert payload.get("stream") is not True

    @pytest.mark.asyncio
    async def test_complete_sends_authorization_header(self) -> None:
        """Authorization ヘッダーが正しく設定されること。"""
        provider = OpenAIProvider(api_key="sk-test-123")

        assert provider._client.headers["authorization"] == "Bearer sk-test-123"

    @pytest.mark.asyncio
    async def test_complete_parses_usage(self, provider: OpenAIProvider) -> None:
        """usage が正しくパースされること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")], model="gpt-4o"
            )

        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 300

    @pytest.mark.asyncio
    async def test_complete_raises_on_http_error(self, provider: OpenAIProvider) -> None:
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
                await provider.complete([ChatMessage(role=Role.user, content="Hi")], model="gpt-4o")

    @pytest.mark.asyncio
    async def test_complete_forwards_kwargs(self, provider: OpenAIProvider) -> None:
        """kwargs が正しくペイロードに伝播されること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")],
                model="gpt-4o",
                temperature=0.5,
                max_tokens=100,
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100

    def test_default_base_url(self) -> None:
        """デフォルトの base_url が OpenAI API であること。"""
        provider = OpenAIProvider(api_key="test-key")

        assert str(provider._client.base_url) == "https://api.openai.com/v1/"

    def test_custom_base_url(self) -> None:
        """カスタム base_url を指定できること。"""
        provider = OpenAIProvider(api_key="test-key", base_url="https://custom.api.com/v1")

        assert str(provider._client.base_url) == "https://custom.api.com/v1/"


class TestOpenAIProviderStream:
    """OpenAIProvider.stream() のテスト。"""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider: OpenAIProvider) -> None:
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
            chunks: list[LLMChunk] = []
            async for chunk in provider.stream(
                [ChatMessage(role=Role.user, content="Hi")], model="gpt-4o"
            ):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self, provider: OpenAIProvider) -> None:
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
            chunks: list[LLMChunk] = []
            async for chunk in provider.stream(
                [ChatMessage(role=Role.user, content="Hi")], model="gpt-4o"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hi"

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self, provider: OpenAIProvider) -> None:
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
            chunks: list[LLMChunk] = []
            async for chunk in provider.stream(
                [ChatMessage(role=Role.user, content="Hi")], model="gpt-4o"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"

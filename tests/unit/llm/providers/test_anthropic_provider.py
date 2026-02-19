"""Tests for Anthropic LLM provider implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.providers.base import (
    ChatMessage,
    LLMChunk,
    LLMProvider,
    LLMResponse,
    Role,
)


class TestAnthropicProviderProtocol:
    """AnthropicProvider が LLMProvider Protocol を満たすこと。"""

    def test_anthropic_provider_satisfies_protocol(self) -> None:
        """AnthropicProvider が LLMProvider Protocol を満たすこと。"""
        provider = AnthropicProvider(api_key="test-key")

        assert isinstance(provider, LLMProvider)


class TestAnthropicProviderComplete:
    """AnthropicProvider.complete() のテスト。"""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        return AnthropicProvider(api_key="test-api-key")

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        """httpx.Response のモックを作成。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, provider: AnthropicProvider) -> None:
        """正常なレスポンスから LLMResponse が返ること。"""
        response_data = {
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [ChatMessage(role=Role.user, content="Hi")]
            result = await provider.complete(messages, model="claude-sonnet-4-20250514")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_complete_maps_usage_fields(self, provider: AnthropicProvider) -> None:
        """Anthropic の usage フィールドが TokenUsage に正しくマッピングされること。"""
        response_data = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 200},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")], model="claude-sonnet-4-20250514"
            )

        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 300

    @pytest.mark.asyncio
    async def test_complete_extracts_no_system_messages(self, provider: AnthropicProvider) -> None:
        """system メッセージがない場合、system パラメータが送信されないこと。"""
        response_data = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [ChatMessage(role=Role.user, content="Hi")]
            await provider.complete(messages, model="claude-sonnet-4-20250514")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "system" not in payload
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_complete_extracts_single_system_message(
        self, provider: AnthropicProvider
    ) -> None:
        """単一の system メッセージが top-level system パラメータに抽出されること。"""
        response_data = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [
                ChatMessage(role=Role.system, content="You are helpful."),
                ChatMessage(role=Role.user, content="Hi"),
            ]
            await provider.complete(messages, model="claude-sonnet-4-20250514")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["system"] == "You are helpful."
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_complete_extracts_multiple_system_messages(
        self, provider: AnthropicProvider
    ) -> None:
        """複数の system メッセージが改行で結合されて top-level system に設定されること。"""
        response_data = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [
                ChatMessage(role=Role.system, content="You are helpful."),
                ChatMessage(role=Role.system, content="Be concise."),
                ChatMessage(role=Role.user, content="Hi"),
            ]
            await provider.complete(messages, model="claude-sonnet-4-20250514")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["system"] == "You are helpful.\n\nBe concise."
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_complete_sends_anthropic_headers(self) -> None:
        """x-api-key と anthropic-version ヘッダーが正しく設定されること。"""
        provider = AnthropicProvider(api_key="sk-ant-test-123")

        assert provider._client.headers["x-api-key"] == "sk-ant-test-123"
        assert provider._client.headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_complete_does_not_use_bearer_auth(self) -> None:
        """Authorization: Bearer ヘッダーが使われていないこと。"""
        provider = AnthropicProvider(api_key="sk-ant-test-123")

        assert "authorization" not in provider._client.headers

    @pytest.mark.asyncio
    async def test_complete_raises_on_http_error(self, provider: AnthropicProvider) -> None:
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
                await provider.complete(
                    [ChatMessage(role=Role.user, content="Hi")],
                    model="claude-sonnet-4-20250514",
                )

    @pytest.mark.asyncio
    async def test_complete_defaults_max_tokens_to_1024(self, provider: AnthropicProvider) -> None:
        """max_tokens が指定されない場合、デフォルト 1024 が使用されること。"""
        response_data = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")], model="claude-sonnet-4-20250514"
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_complete_forwards_kwargs(self, provider: AnthropicProvider) -> None:
        """kwargs が正しくペイロードに伝播されること。"""
        response_data = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")],
                model="claude-sonnet-4-20250514",
                temperature=0.7,
                max_tokens=2048,
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_complete_stop_reason_passthrough(self, provider: AnthropicProvider) -> None:
        """stop_reason が finish_reason としてパススルーされること。"""
        response_data = {
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")], model="claude-sonnet-4-20250514"
            )

        assert result.finish_reason == "max_tokens"


class TestAnthropicProviderStream:
    """AnthropicProvider.stream() のテスト。"""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        return AnthropicProvider(api_key="test-api-key")

    @staticmethod
    def _delta_line(text: str) -> bytes:
        """content_block_delta イベントの data 行を生成するヘルパー。"""
        return (
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"' + text.encode() + b'"}}'
        )

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider: AnthropicProvider) -> None:
        """ストリーミングで content_block_delta から LLMChunk が yield されること。"""
        lines = [
            b"event: message_start",
            b'data: {"type":"message_start","message":{"model":"test"}}',
            b"event: content_block_start",
            b'data: {"type":"content_block_start","index":0}',
            b"event: content_block_delta",
            self._delta_line("Hello"),
            b"event: content_block_delta",
            self._delta_line(" world"),
            b"event: message_stop",
            b'data: {"type":"message_stop"}',
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
                [ChatMessage(role=Role.user, content="Hi")], model="claude-sonnet-4-20250514"
            ):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    @pytest.mark.asyncio
    async def test_stream_stops_at_message_stop(self, provider: AnthropicProvider) -> None:
        """message_stop でストリーミングが終了すること。"""
        lines = [
            b"event: content_block_delta",
            self._delta_line("Hi"),
            b"event: message_stop",
            b'data: {"type":"message_stop"}',
            b"event: content_block_delta",
            self._delta_line("should not appear"),
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
                [ChatMessage(role=Role.user, content="Hi")], model="claude-sonnet-4-20250514"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hi"

    @pytest.mark.asyncio
    async def test_stream_skips_non_text_delta_events(self, provider: AnthropicProvider) -> None:
        """text_delta 以外の delta タイプはスキップされること。"""
        json_delta = (
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"input_json_delta","partial_json":"{}"}}'
        )
        lines = [
            b"event: content_block_delta",
            json_delta,
            b"event: content_block_delta",
            self._delta_line("Hello"),
            b"event: message_stop",
            b'data: {"type":"message_stop"}',
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
                [ChatMessage(role=Role.user, content="Hi")], model="claude-sonnet-4-20250514"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"

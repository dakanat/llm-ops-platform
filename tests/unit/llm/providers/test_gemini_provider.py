"""Tests for Gemini LLM provider implementation."""

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
from src.llm.providers.gemini_provider import GeminiProvider


class TestGeminiProviderProtocol:
    """GeminiProvider が LLMProvider Protocol を満たすこと。"""

    def test_gemini_provider_satisfies_protocol(self) -> None:
        """GeminiProvider が LLMProvider Protocol を満たすこと。"""
        provider = GeminiProvider(api_key="test-key")

        assert isinstance(provider, LLMProvider)


class TestGeminiProviderComplete:
    """GeminiProvider.complete() のテスト。"""

    @pytest.fixture
    def provider(self) -> GeminiProvider:
        return GeminiProvider(api_key="test-api-key")

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        """httpx.Response のモックを作成。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({})
        return mock_resp

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, provider: GeminiProvider) -> None:
        """正常なレスポンスから LLMResponse が返ること。"""
        response_data = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "model": "gemini-2.5-flash-lite",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [ChatMessage(role=Role.user, content="Hi")]
            result = await provider.complete(messages, model="gemini-2.5-flash-lite")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.model == "gemini-2.5-flash-lite"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_sends_correct_payload(self, provider: GeminiProvider) -> None:
        """正しいペイロードが送信されること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "gemini-2.5-flash-lite",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            messages = [
                ChatMessage(role=Role.system, content="You are helpful."),
                ChatMessage(role=Role.user, content="Hi"),
            ]
            await provider.complete(messages, model="gemini-2.5-flash-lite")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "gemini-2.5-flash-lite"
        assert payload["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        assert payload.get("stream") is not True

    @pytest.mark.asyncio
    async def test_complete_sends_authorization_header(self) -> None:
        """Authorization ヘッダーが正しく設定されること。"""
        provider = GeminiProvider(api_key="gemini-test-123")

        assert provider._client.headers["authorization"] == "Bearer gemini-test-123"

    @pytest.mark.asyncio
    async def test_complete_parses_usage(self, provider: GeminiProvider) -> None:
        """usage が正しくパースされること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "gemini-2.5-flash-lite",
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")], model="gemini-2.5-flash-lite"
            )

        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 300

    @pytest.mark.asyncio
    async def test_complete_raises_on_http_error(self, provider: GeminiProvider) -> None:
        """HTTPエラー時に例外が発生すること (400 はリトライせず即座に raise)。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.headers = httpx.Headers({})
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(spec=httpx.Request),
            response=mock_resp,
        )

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp

            with pytest.raises(httpx.HTTPStatusError):
                await provider.complete(
                    [ChatMessage(role=Role.user, content="Hi")], model="gemini-2.5-flash-lite"
                )

    @pytest.mark.asyncio
    async def test_complete_forwards_kwargs(self, provider: GeminiProvider) -> None:
        """kwargs が正しくペイロードに伝播されること。"""
        response_data = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "model": "gemini-2.5-flash-lite",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await provider.complete(
                [ChatMessage(role=Role.user, content="Hi")],
                model="gemini-2.5-flash-lite",
                temperature=0.5,
                max_tokens=100,
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100

    def test_default_base_url(self) -> None:
        """デフォルトの base_url が Gemini OpenAI 互換エンドポイントであること。"""
        provider = GeminiProvider(api_key="test-key")

        assert (
            str(provider._client.base_url)
            == "https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def test_custom_base_url(self) -> None:
        """カスタム base_url を指定できること。"""
        provider = GeminiProvider(api_key="test-key", base_url="https://custom.api.com/v1")

        assert str(provider._client.base_url) == "https://custom.api.com/v1/"


class TestGeminiProviderStream:
    """GeminiProvider.stream() のテスト。"""

    @pytest.fixture
    def provider(self) -> GeminiProvider:
        return GeminiProvider(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider: GeminiProvider) -> None:
        """ストリーミングで LLMChunk が yield されること。"""
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
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
                [ChatMessage(role=Role.user, content="Hi")], model="gemini-2.5-flash-lite"
            ):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self, provider: GeminiProvider) -> None:
        """data: [DONE] でストリーミングが終了すること。"""
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
            b"data: [DONE]",
            b'data: {"choices":[{"delta":{"content":"should not appear"},"finish_reason":null}]}',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
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
                [ChatMessage(role=Role.user, content="Hi")], model="gemini-2.5-flash-lite"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hi"

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self, provider: GeminiProvider) -> None:
        """空の content を持つ delta はスキップされること。"""
        lines = [
            b'data: {"choices":[{"delta":{"content":""},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":null}]}',
            b"data: [DONE]",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
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
                [ChatMessage(role=Role.user, content="Hi")], model="gemini-2.5-flash-lite"
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"


def _make_response(
    status_code: int,
    json_data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """httpx.Response モックを生成する。"""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.headers = httpx.Headers(headers or {})
    if json_data is not None:
        mock_resp.json.return_value = json_data
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code} Error",
            request=MagicMock(spec=httpx.Request),
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


_SUCCESS_DATA: dict[str, Any] = {
    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
    "model": "gemini-2.5-flash-lite",
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
}

_MESSAGES = [ChatMessage(role=Role.user, content="Hi")]


class TestGeminiRetryOnRateLimit:
    """GeminiProvider の 429/5xx リトライ動作を検証する。"""

    @pytest.fixture()
    def provider(self) -> GeminiProvider:
        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio()
    async def test_retries_on_429_and_succeeds(self, provider: GeminiProvider) -> None:
        """429 を2回受け取った後に200で成功すること。"""
        responses = [
            _make_response(429),
            _make_response(429),
            _make_response(200, json_data=_SUCCESS_DATA),
        ]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.gemini_provider.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            result = await provider.complete(_MESSAGES, model="gemini-2.5-flash-lite")

        assert isinstance(result, LLMResponse)
        assert result.content == "OK"
        assert mock_sleep.await_count == 2

    @pytest.mark.asyncio()
    async def test_exhausts_max_retries_and_raises(self, provider: GeminiProvider) -> None:
        """全試行が429の場合に HTTPStatusError が送出されること。"""
        responses = [_make_response(429) for _ in range(6)]  # _MAX_RETRIES=5 → 6 attempts

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch("src.llm.providers.gemini_provider.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await provider.complete(_MESSAGES, model="gemini-2.5-flash-lite")

    @pytest.mark.asyncio()
    async def test_no_retry_on_400(self, provider: GeminiProvider) -> None:
        """400 はリトライせず即座に raise すること。"""
        responses = [_make_response(400)]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.gemini_provider.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
            pytest.raises(httpx.HTTPStatusError),
        ):
            await provider.complete(_MESSAGES, model="gemini-2.5-flash-lite")

        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_retries_on_500(self, provider: GeminiProvider) -> None:
        """500 もリトライ対象であること。"""
        responses = [
            _make_response(500),
            _make_response(200, json_data=_SUCCESS_DATA),
        ]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.gemini_provider.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            result = await provider.complete(_MESSAGES, model="gemini-2.5-flash-lite")

        assert result.content == "OK"
        assert mock_sleep.await_count == 1

    @pytest.mark.asyncio()
    async def test_respects_retry_after_header(self, provider: GeminiProvider) -> None:
        """Retry-After ヘッダーの値を sleep に渡すこと。"""
        responses = [
            _make_response(429, headers={"retry-after": "3"}),
            _make_response(200, json_data=_SUCCESS_DATA),
        ]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.gemini_provider.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            await provider.complete(_MESSAGES, model="gemini-2.5-flash-lite")

        mock_sleep.assert_awaited_once_with(3.0)

    @pytest.mark.asyncio()
    async def test_stream_retries_on_429(self, provider: GeminiProvider) -> None:
        """stream() でも 429 リトライが動作すること。"""
        # 1回目: 429 レスポンス
        mock_429_response = MagicMock()
        mock_429_response.status_code = 429
        mock_429_response.headers = httpx.Headers({})
        mock_429_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Error",
            request=MagicMock(spec=httpx.Request),
            response=mock_429_response,
        )

        mock_429_cm = AsyncMock()
        mock_429_cm.__aenter__ = AsyncMock(return_value=mock_429_response)
        mock_429_cm.__aexit__ = AsyncMock(return_value=False)

        # 2回目: 200 レスポンス（ストリーム成功）
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            b"data: [DONE]",
        ]

        mock_200_response = MagicMock()
        mock_200_response.status_code = 200
        mock_200_response.raise_for_status = MagicMock()

        async def mock_aiter_lines() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield line

        mock_200_response.aiter_lines = mock_aiter_lines

        mock_200_cm = AsyncMock()
        mock_200_cm.__aenter__ = AsyncMock(return_value=mock_200_response)
        mock_200_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(
                provider._client,
                "stream",
                side_effect=[mock_429_cm, mock_200_cm],
            ),
            patch(
                "src.llm.providers.gemini_provider.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            chunks = []
            async for chunk in provider.stream(_MESSAGES, model="gemini-2.5-flash-lite"):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
        assert mock_sleep.await_count == 1

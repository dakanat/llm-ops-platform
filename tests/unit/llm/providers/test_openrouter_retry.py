"""Tests for OpenRouterProvider retry logic on 429 and 5xx responses."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.llm.providers.base import ChatMessage, LLMResponse, Role
from src.llm.providers.openrouter import OpenRouterProvider


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
    "model": "test-model",
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
}

_MESSAGES = [ChatMessage(role=Role.user, content="Hi")]


class TestOpenRouterRetryOnRateLimit:
    """OpenRouterProvider の 429/5xx リトライ動作を検証する。"""

    @pytest.fixture()
    def provider(self) -> OpenRouterProvider:
        return OpenRouterProvider(api_key="test-key")

    @pytest.mark.asyncio()
    async def test_retries_on_429_and_succeeds(self, provider: OpenRouterProvider) -> None:
        """429 を2回受け取った後に200で成功すること。"""
        responses = [
            _make_response(429),
            _make_response(429),
            _make_response(200, json_data=_SUCCESS_DATA),
        ]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.openrouter.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            result = await provider.complete(_MESSAGES, model="test-model")

        assert isinstance(result, LLMResponse)
        assert result.content == "OK"
        assert mock_sleep.await_count == 2

    @pytest.mark.asyncio()
    async def test_exhausts_max_retries_and_raises(self, provider: OpenRouterProvider) -> None:
        """全試行が429の場合に HTTPStatusError が送出されること。"""
        responses = [_make_response(429) for _ in range(4)]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch("src.llm.providers.openrouter.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await provider.complete(_MESSAGES, model="test-model")

    @pytest.mark.asyncio()
    async def test_no_retry_on_400(self, provider: OpenRouterProvider) -> None:
        """400 はリトライせず即座に raise すること。"""
        responses = [_make_response(400)]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.openrouter.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
            pytest.raises(httpx.HTTPStatusError),
        ):
            await provider.complete(_MESSAGES, model="test-model")

        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_retries_on_500(self, provider: OpenRouterProvider) -> None:
        """500 もリトライ対象であること。"""
        responses = [
            _make_response(500),
            _make_response(200, json_data=_SUCCESS_DATA),
        ]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.openrouter.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            result = await provider.complete(_MESSAGES, model="test-model")

        assert result.content == "OK"
        assert mock_sleep.await_count == 1

    @pytest.mark.asyncio()
    async def test_respects_retry_after_header(self, provider: OpenRouterProvider) -> None:
        """Retry-After ヘッダーの値を sleep に渡すこと。"""
        responses = [
            _make_response(429, headers={"retry-after": "2"}),
            _make_response(200, json_data=_SUCCESS_DATA),
        ]

        with (
            patch.object(provider._client, "post", new_callable=AsyncMock, side_effect=responses),
            patch(
                "src.llm.providers.openrouter.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            await provider.complete(_MESSAGES, model="test-model")

        mock_sleep.assert_awaited_once_with(2.0)

    @pytest.mark.asyncio()
    async def test_stream_retries_on_429(self, provider: OpenRouterProvider) -> None:
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
                "src.llm.providers.openrouter.asyncio.sleep", new_callable=AsyncMock
            ) as mock_sleep,
        ):
            chunks = []
            async for chunk in provider.stream(_MESSAGES, model="test-model"):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
        assert mock_sleep.await_count == 1

"""Tests for POST /chat endpoint (non-streaming and streaming)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

from httpx import AsyncClient
from src.llm.providers.base import (
    ChatMessage,
    LLMChunk,
    LLMResponse,
    Role,
    TokenUsage,
)
from src.main import app


def _mock_provider(
    response: LLMResponse | None = None,
    chunks: list[LLMChunk] | None = None,
) -> AsyncMock:
    """Create a mock LLM provider with optional response/chunks."""
    provider = AsyncMock()
    provider.complete.return_value = response or LLMResponse(
        content="Hello!",
        model="test-model",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        finish_reason="stop",
    )

    async def _stream(*args: object, **kwargs: object) -> AsyncGenerator[LLMChunk, None]:
        for chunk in chunks or [LLMChunk(content="Hello!", finish_reason="stop")]:
            yield chunk

    provider.stream = _stream
    return provider


def _override_dependencies(
    provider: AsyncMock | None = None,
    model: str = "test-model",
) -> None:
    """Set FastAPI dependency overrides for chat route tests."""
    from src.api.dependencies import get_llm_model, get_llm_provider

    mock_provider = provider or _mock_provider()
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_llm_model] = lambda: model


# =============================================================================
# Non-streaming
# =============================================================================


class TestChatRouteNonStreaming:
    """POST /chat (stream=false) のテスト。"""

    async def test_returns_200_with_valid_request(self, client: AsyncClient) -> None:
        """有効なリクエストで 200 が返ること。"""
        _override_dependencies()
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )

        assert response.status_code == 200

    async def test_returns_content_from_llm_provider(self, client: AsyncClient) -> None:
        """LLM プロバイダの応答内容がレスポンスに含まれること。"""
        provider = _mock_provider(response=LLMResponse(content="Hi there!", model="test-model"))
        _override_dependencies(provider=provider)
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )

        assert response.json()["content"] == "Hi there!"

    async def test_returns_model_name(self, client: AsyncClient) -> None:
        """レスポンスに model フィールドが含まれること。"""
        _override_dependencies(model="openai/gpt-oss-120b:free")
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )

        assert response.json()["model"] == "openai/gpt-oss-120b:free"

    async def test_returns_usage_when_available(self, client: AsyncClient) -> None:
        """usage が利用可能な場合にレスポンスに含まれること。"""
        provider = _mock_provider(
            response=LLMResponse(
                content="Hi",
                model="test-model",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            )
        )
        _override_dependencies(provider=provider)
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )

        usage = response.json()["usage"]
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 10
        assert usage["total_tokens"] == 15

    async def test_returns_null_usage_when_not_available(self, client: AsyncClient) -> None:
        """usage が利用不可の場合に null が返ること。"""
        provider = _mock_provider(
            response=LLMResponse(content="Hi", model="test-model", usage=None)
        )
        _override_dependencies(provider=provider)
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )

        assert response.json()["usage"] is None

    async def test_converts_messages_to_internal_format(self, client: AsyncClient) -> None:
        """リクエストのメッセージが ChatMessage に変換されてプロバイダに渡されること。"""
        provider = _mock_provider()
        _override_dependencies(provider=provider)
        await client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                "stream": False,
            },
        )

        call_args = provider.complete.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert isinstance(messages[0], ChatMessage)
        assert messages[0].role == Role.system
        assert messages[1].role == Role.user

    async def test_returns_422_for_invalid_role(self, client: AsyncClient) -> None:
        """無効な role で 422 が返ること。"""
        _override_dependencies()
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "invalid", "content": "Hello"}], "stream": False},
        )

        assert response.status_code == 422

    async def test_returns_422_for_missing_messages(self, client: AsyncClient) -> None:
        """messages フィールドがない場合に 422 が返ること。"""
        _override_dependencies()
        response = await client.post("/chat", json={"stream": False})

        assert response.status_code == 422

    async def test_returns_502_when_llm_provider_fails(self, client: AsyncClient) -> None:
        """LLM プロバイダがエラーの場合に 502 が返ること。"""
        provider = _mock_provider()
        provider.complete.side_effect = RuntimeError("LLM failed")
        _override_dependencies(provider=provider)
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": False},
        )

        assert response.status_code == 502

    async def test_defaults_stream_to_false(self, client: AsyncClient) -> None:
        """stream を省略した場合に JSON レスポンスが返ること。"""
        _override_dependencies()
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


# =============================================================================
# Streaming
# =============================================================================


class TestChatRouteStreaming:
    """POST /chat (stream=true) のテスト。"""

    async def test_returns_200_with_event_stream_content_type(self, client: AsyncClient) -> None:
        """ストリーミング時に text/event-stream Content-Type が返ること。"""
        _override_dependencies()
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    async def test_yields_sse_formatted_chunks(self, client: AsyncClient) -> None:
        """SSE 形式 (data: {...}\\n\\n) のチャンクが返ること。"""
        chunks = [
            LLMChunk(content="Hi", finish_reason=None),
            LLMChunk(content=" there!", finish_reason="stop"),
        ]
        provider = _mock_provider(chunks=chunks)
        _override_dependencies(provider=provider)
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

        lines = response.text.strip().split("\n\n")
        for line in lines[:-1]:  # Exclude [DONE]
            assert line.startswith("data: ")

    async def test_ends_with_done(self, client: AsyncClient) -> None:
        """最後のイベントが data: [DONE] であること。"""
        _override_dependencies()
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

        assert response.text.rstrip().endswith("data: [DONE]")

    async def test_chunk_content_matches_provider_output(self, client: AsyncClient) -> None:
        """チャンクの content がプロバイダの出力と一致すること。"""
        chunks = [
            LLMChunk(content="Hi", finish_reason=None),
            LLMChunk(content=" there!", finish_reason="stop"),
        ]
        provider = _mock_provider(chunks=chunks)
        _override_dependencies(provider=provider)
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

        events = response.text.strip().split("\n\n")
        # Parse content chunks (excluding [DONE])
        data_events = [e for e in events if e.startswith("data: ") and "[DONE]" not in e]
        parsed = [json.loads(e.removeprefix("data: ")) for e in data_events]
        assert parsed[0]["content"] == "Hi"
        assert parsed[1]["content"] == " there!"

    async def test_returns_422_for_invalid_request(self, client: AsyncClient) -> None:
        """ストリーミングでも無効なリクエストで 422 が返ること。"""
        _override_dependencies()
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "invalid", "content": "Hello"}], "stream": True},
        )

        assert response.status_code == 422

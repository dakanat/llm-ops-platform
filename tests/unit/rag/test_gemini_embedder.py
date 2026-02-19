"""Tests for GeminiEmbedder (Gemini native Embedding API)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from src.rag.embedder import EmbedderProtocol, EmbeddingError
from src.rag.gemini_embedder import GeminiEmbedder

_DIMENSIONS = 1024


def _make_embed_response(dimensions: int = _DIMENSIONS) -> dict[str, Any]:
    """embedContent API の正常レスポンスを生成。"""
    return {"embedding": {"values": [0.1] * dimensions}}


def _make_batch_response(count: int, dimensions: int = _DIMENSIONS) -> dict[str, Any]:
    """batchEmbedContents API の正常レスポンスを生成。"""
    return {"embeddings": [{"values": [0.1 * (i + 1)] * dimensions} for i in range(count)]}


def _make_httpx_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """テスト用の httpx.Response を生成。"""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        headers=headers or {},
        request=httpx.Request("POST", "https://example.com"),
    )
    return resp


class TestGeminiEmbedderEmbed:
    """GeminiEmbedder.embed() 単一テキストのテスト。"""

    @pytest.fixture
    def embedder(self) -> GeminiEmbedder:
        return GeminiEmbedder(api_key="test-key", model="gemini-embedding-001")

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_floats(self, embedder: GeminiEmbedder) -> None:
        """単一テキストから list[float] が返ること。"""
        mock_resp = _make_httpx_response(json_data=_make_embed_response())

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed("テスト文章")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_returns_1024_dimensions(self, embedder: GeminiEmbedder) -> None:
        """outputDimensionality=1024 により 1024 次元が返ること。"""
        mock_resp = _make_httpx_response(json_data=_make_embed_response(dimensions=1024))

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed("テスト文章")

        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_embed_sends_correct_url_and_payload(self, embedder: GeminiEmbedder) -> None:
        """URL が models/...:embedContent で payload に outputDimensionality が含まれること。"""
        mock_resp = _make_httpx_response(json_data=_make_embed_response())

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await embedder.embed("テスト文章")

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "models/gemini-embedding-001:embedContent" in str(url)

        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["content"]["parts"][0]["text"] == "テスト文章"
        assert payload["outputDimensionality"] == 1024

    @pytest.mark.asyncio
    async def test_embed_sends_api_key_as_query_param(self, embedder: GeminiEmbedder) -> None:
        """API キーがクエリパラメータ ?key=... で送信されること。"""
        mock_resp = _make_httpx_response(json_data=_make_embed_response())

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await embedder.embed("テスト文章")

        call_args = mock_post.call_args
        params = call_args.kwargs.get("params", {})
        assert params.get("key") == "test-key"

    @pytest.mark.asyncio
    async def test_embed_raises_on_empty_string(self, embedder: GeminiEmbedder) -> None:
        """空文字で EmbeddingError が発生すること。"""
        with pytest.raises(EmbeddingError):
            await embedder.embed("")

    @pytest.mark.asyncio
    async def test_embed_raises_on_http_error(self, embedder: GeminiEmbedder) -> None:
        """HTTP 400 エラーで EmbeddingError が発生すること。"""
        mock_resp = _make_httpx_response(status_code=400, json_data={"error": "bad"})

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            with pytest.raises(EmbeddingError):
                await embedder.embed("テスト文章")

    @pytest.mark.asyncio
    async def test_embed_raises_on_connection_error(self, embedder: GeminiEmbedder) -> None:
        """接続エラーで EmbeddingError が発生すること。"""
        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(EmbeddingError):
                await embedder.embed("テスト文章")


class TestGeminiEmbedderEmbedBatch:
    """GeminiEmbedder.embed_batch() バッチのテスト。"""

    @pytest.fixture
    def embedder(self) -> GeminiEmbedder:
        return GeminiEmbedder(api_key="test-key", model="gemini-embedding-001")

    @pytest.mark.asyncio
    async def test_embed_batch_returns_list_of_vectors(self, embedder: GeminiEmbedder) -> None:
        """複数テキストから list[list[float]] が返ること。"""
        mock_resp = _make_httpx_response(json_data=_make_batch_response(count=3))

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed_batch(["a", "b", "c"])

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(f, float) for v in result for f in v)

    @pytest.mark.asyncio
    async def test_embed_batch_uses_batch_endpoint(self, embedder: GeminiEmbedder) -> None:
        """URL が models/...:batchEmbedContents であること。"""
        mock_resp = _make_httpx_response(json_data=_make_batch_response(count=2))

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await embedder.embed_batch(["a", "b"])

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "models/gemini-embedding-001:batchEmbedContents" in str(url)

    @pytest.mark.asyncio
    async def test_embed_batch_raises_on_empty_list(self, embedder: GeminiEmbedder) -> None:
        """空リストで EmbeddingError が発生すること。"""
        with pytest.raises(EmbeddingError):
            await embedder.embed_batch([])

    @pytest.mark.asyncio
    async def test_embed_batch_raises_on_any_empty_string(self, embedder: GeminiEmbedder) -> None:
        """空文字を含むリストで EmbeddingError が発生すること。"""
        with pytest.raises(EmbeddingError):
            await embedder.embed_batch(["valid", "", "also valid"])

    @pytest.mark.asyncio
    async def test_embed_batch_splits_at_100_items(self, embedder: GeminiEmbedder) -> None:
        """150 件 → 2 回 HTTP コール (100 + 50) に自動分割されること。"""
        resp_100 = _make_httpx_response(json_data=_make_batch_response(count=100))
        resp_50 = _make_httpx_response(json_data=_make_batch_response(count=50))

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [resp_100, resp_50]
            texts = [f"text_{i}" for i in range(150)]
            result = await embedder.embed_batch(texts)

        assert mock_post.call_count == 2
        assert len(result) == 150


class TestGeminiEmbedderRetry:
    """GeminiEmbedder のリトライ機能のテスト。"""

    @pytest.fixture
    def embedder(self) -> GeminiEmbedder:
        return GeminiEmbedder(api_key="test-key", model="gemini-embedding-001")

    @pytest.mark.asyncio
    async def test_retries_on_429_and_succeeds(self, embedder: GeminiEmbedder) -> None:
        """429 → リトライ → 成功すること。"""
        resp_429 = _make_httpx_response(status_code=429, json_data={"error": "rate"})
        resp_ok = _make_httpx_response(json_data=_make_embed_response())

        with (
            patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post,
            patch("src.rag.gemini_embedder.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_post.side_effect = [resp_429, resp_ok]
            result = await embedder.embed("テスト")

        assert isinstance(result, list)
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self, embedder: GeminiEmbedder) -> None:
        """全て 429 でリトライ上限到達時に EmbeddingError が発生すること。"""
        resp_429 = _make_httpx_response(status_code=429, json_data={"error": "rate"})

        with (
            patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post,
            patch("src.rag.gemini_embedder.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_post.return_value = resp_429
            with pytest.raises(EmbeddingError):
                await embedder.embed("テスト")

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self, embedder: GeminiEmbedder) -> None:
        """400 エラーでリトライせず即座に EmbeddingError が発生すること。"""
        resp_400 = _make_httpx_response(status_code=400, json_data={"error": "bad"})

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = resp_400
            with pytest.raises(EmbeddingError):
                await embedder.embed("テスト")

        assert mock_post.call_count == 1


class TestGeminiEmbedderLifecycle:
    """GeminiEmbedder のライフサイクルテスト。"""

    @pytest.mark.asyncio
    async def test_close_closes_http_client(self) -> None:
        """close() で内部 HTTP クライアントの aclose() が呼ばれること。"""
        embedder = GeminiEmbedder(api_key="test-key", model="gemini-embedding-001")

        with patch.object(embedder._client, "aclose", new_callable=AsyncMock) as mock_aclose:
            await embedder.close()

        mock_aclose.assert_awaited_once()

    def test_gemini_embedder_satisfies_protocol(self) -> None:
        """GeminiEmbedder が EmbedderProtocol を満たすこと。"""
        embedder = GeminiEmbedder(api_key="test-key", model="gemini-embedding-001")
        assert isinstance(embedder, EmbedderProtocol)

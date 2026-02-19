"""Tests for Embedding client (vLLM local server)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from src.rag.embedder import Embedder, EmbedderProtocol, EmbeddingError


def _make_embedding(dimensions: int = 1024, index: int = 0) -> dict[str, Any]:
    """テスト用の embedding レスポンスオブジェクトを生成。"""
    return {"embedding": [0.1 * (index + 1)] * dimensions, "index": index}


class TestEmbedderProtocol:
    """Embedder が EmbedderProtocol を満たすことのテスト。"""

    def test_embedder_satisfies_protocol(self) -> None:
        """Embedder が EmbedderProtocol の runtime_checkable を満たすこと。"""
        embedder = Embedder(base_url="http://localhost:8001/v1", model="test-model")
        assert isinstance(embedder, EmbedderProtocol)


class TestEmbedderEmbed:
    """Embedder.embed() 単一テキストのテスト。"""

    @pytest.fixture
    def embedder(self) -> Embedder:
        return Embedder(base_url="http://localhost:8001/v1", model="cl-nagoya/ruri-v3-310m")

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        """httpx.Response のモックを作成。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_floats(self, embedder: Embedder) -> None:
        """テキストから list[float] が返ること。"""
        response_data = {"data": [_make_embedding()]}
        mock_resp = self._mock_response(response_data)

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed("テスト文章")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_returns_1024_dimensions(self, embedder: Embedder) -> None:
        """次元数が 1024 であること。"""
        response_data = {"data": [_make_embedding(dimensions=1024)]}
        mock_resp = self._mock_response(response_data)

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed("テスト文章")

        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_embed_sends_correct_payload(self, embedder: Embedder) -> None:
        """POST に model と input が含まれること。"""
        response_data = {"data": [_make_embedding()]}
        mock_resp = self._mock_response(response_data)

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            await embedder.embed("テスト文章")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "cl-nagoya/ruri-v3-310m"
        assert payload["input"] == ["テスト文章"]

    @pytest.mark.asyncio
    async def test_embed_raises_on_empty_string(self, embedder: Embedder) -> None:
        """空文字で EmbeddingError が発生すること。"""
        with pytest.raises(EmbeddingError):
            await embedder.embed("")

    @pytest.mark.asyncio
    async def test_embed_raises_on_http_error(self, embedder: Embedder) -> None:
        """HTTP エラーで EmbeddingError が発生すること。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(spec=httpx.Request),
            response=MagicMock(spec=httpx.Response),
        )

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            with pytest.raises(EmbeddingError) as exc_info:
                await embedder.embed("テスト文章")

        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_embed_raises_on_connection_error(self, embedder: Embedder) -> None:
        """接続エラーで EmbeddingError が発生すること。"""
        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(EmbeddingError) as exc_info:
                await embedder.embed("テスト文章")

        assert exc_info.value.__cause__ is not None


class TestEmbedderEmbedBatch:
    """Embedder.embed_batch() バッチのテスト。"""

    @pytest.fixture
    def embedder(self) -> Embedder:
        return Embedder(base_url="http://localhost:8001/v1", model="cl-nagoya/ruri-v3-310m")

    def _mock_response(self, data: dict[str, Any]) -> MagicMock:
        """httpx.Response のモックを作成。"""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @pytest.mark.asyncio
    async def test_embed_batch_returns_list_of_vectors(self, embedder: Embedder) -> None:
        """複数テキストから list[list[float]] が返ること。"""
        response_data = {
            "data": [_make_embedding(index=0), _make_embedding(index=1)],
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed_batch(["テスト1", "テスト2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(f, float) for v in result for f in v)

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self, embedder: Embedder) -> None:
        """index でソートし入力順を保持すること。"""
        # API が逆順で返しても正しい順序になること
        response_data = {
            "data": [
                _make_embedding(index=1),
                _make_embedding(index=0),
            ],
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed_batch(["テスト0", "テスト1"])

        # index=0 の embedding は 0.1 * 1 = 0.1, index=1 は 0.1 * 2 = 0.2
        assert result[0][0] == pytest.approx(0.1)
        assert result[1][0] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_embed_batch_each_vector_has_1024_dimensions(self, embedder: Embedder) -> None:
        """各ベクトルが 1024 次元であること。"""
        response_data = {
            "data": [
                _make_embedding(dimensions=1024, index=0),
                _make_embedding(dimensions=1024, index=1),
                _make_embedding(dimensions=1024, index=2),
            ],
        }
        mock_resp = self._mock_response(response_data)

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await embedder.embed_batch(["a", "b", "c"])

        assert all(len(v) == 1024 for v in result)

    @pytest.mark.asyncio
    async def test_embed_batch_raises_on_empty_list(self, embedder: Embedder) -> None:
        """空リストで EmbeddingError が発生すること。"""
        with pytest.raises(EmbeddingError):
            await embedder.embed_batch([])

    @pytest.mark.asyncio
    async def test_embed_batch_raises_on_any_empty_string(self, embedder: Embedder) -> None:
        """リスト内に空文字が含まれる場合 EmbeddingError が発生すること。"""
        with pytest.raises(EmbeddingError):
            await embedder.embed_batch(["有効なテキスト", "", "別のテキスト"])


class TestEmbedderLifecycle:
    """Embedder の初期化・終了のテスト。"""

    def test_initializes_with_base_url_and_model(self) -> None:
        """base_url と model で初期化できること。"""
        embedder = Embedder(base_url="http://localhost:8001/v1", model="test-model")

        assert embedder._model == "test-model"
        assert str(embedder._client.base_url).rstrip("/") == "http://localhost:8001/v1"

    @pytest.mark.asyncio
    async def test_close_closes_http_client(self) -> None:
        """close() で内部 HTTP クライアントの aclose() が呼ばれること。"""
        embedder = Embedder(base_url="http://localhost:8001/v1", model="test-model")

        with patch.object(embedder._client, "aclose", new_callable=AsyncMock) as mock_aclose:
            await embedder.close()

        mock_aclose.assert_awaited_once()

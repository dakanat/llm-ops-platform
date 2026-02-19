"""Embedding client for Gemini native API (embedContent / batchEmbedContents)."""

import asyncio
from typing import Any

import httpx

from src.rag.embedder import EmbeddingError

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_OUTPUT_DIMENSIONALITY = 1024
_BATCH_LIMIT = 100

_MAX_RETRIES = 5
_BASE_DELAY = 2.0  # seconds
_MAX_DELAY = 60.0  # seconds
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _calc_delay(response: httpx.Response, attempt: int) -> float:
    """Calculate retry delay from Retry-After header or exponential backoff."""
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass
    delay: float = _BASE_DELAY * (2**attempt)
    return min(delay, _MAX_DELAY)


class GeminiEmbedder:
    """Gemini native Embedding API を使用する非同期クライアント。

    ``embedContent`` / ``batchEmbedContents`` エンドポイントで
    Embedding ベクトルを取得する。バッチは 100 件ごとに自動分割する。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._client = httpx.AsyncClient(base_url=self._base_url)

    async def embed(self, text: str) -> list[float]:
        """単一テキストの Embedding ベクトルを取得する。

        Args:
            text: Embedding を生成するテキスト。空文字は不可。

        Returns:
            1024 次元の float リスト。

        Raises:
            EmbeddingError: 空文字、HTTP エラー、接続エラーの場合。
        """
        if not text:
            raise EmbeddingError("text must not be empty")

        url = f"/models/{self._model}:embedContent"
        payload: dict[str, Any] = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": _OUTPUT_DIMENSIONALITY,
        }
        data = await self._request_with_retry(url, payload)
        return data["embedding"]["values"]  # type: ignore[no-any-return]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストの Embedding ベクトルをバッチ取得する。

        100 件を超える場合は自動的に分割してリクエストする。

        Args:
            texts: Embedding を生成するテキストのリスト。空リスト・空文字を含むリストは不可。

        Returns:
            入力順に並んだ 1024 次元の float リストのリスト。

        Raises:
            EmbeddingError: 空リスト、空文字を含む場合、HTTP エラー、接続エラーの場合。
        """
        if not texts:
            raise EmbeddingError("texts must not be empty")
        if any(not t for t in texts):
            raise EmbeddingError("texts must not contain empty strings")

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_LIMIT):
            chunk = texts[i : i + _BATCH_LIMIT]
            vectors = await self._request_batch(chunk)
            all_vectors.extend(vectors)
        return all_vectors

    async def close(self) -> None:
        """内部 HTTP クライアントを閉じる。"""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    async def _request_batch(self, texts: list[str]) -> list[list[float]]:
        """batchEmbedContents エンドポイントにリクエストを送信する。"""
        url = f"/models/{self._model}:batchEmbedContents"
        requests_payload: list[dict[str, Any]] = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": _OUTPUT_DIMENSIONALITY,
            }
            for t in texts
        ]
        payload: dict[str, Any] = {"requests": requests_payload}
        data = await self._request_with_retry(url, payload)
        return [emb["values"] for emb in data["embeddings"]]

    async def _request_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """リトライ付きで POST リクエストを送信する。"""
        last_response: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(url, json=payload, params={"key": self._api_key})
            except httpx.ConnectError as e:
                raise EmbeddingError(f"Connection error: {e}") from e

            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_response = response
                if attempt < _MAX_RETRIES:
                    delay = _calc_delay(response, attempt)
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(
                    f"HTTP error after {_MAX_RETRIES} retries: {response.status_code}"
                )

            if response.status_code >= 400:
                raise EmbeddingError(f"HTTP error: {response.status_code}")

            return response.json()  # type: ignore[no-any-return]

        # Unreachable, but satisfies the type checker.
        assert last_response is not None  # noqa: S101
        raise EmbeddingError(
            f"HTTP error after {_MAX_RETRIES} retries: {last_response.status_code}"
        )

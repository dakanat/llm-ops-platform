"""Embedding client for local vLLM server (OpenAI-compatible API)."""

from __future__ import annotations

import httpx


class EmbeddingError(Exception):
    """Embedding リクエストに関するエラー。"""


class Embedder:
    """vLLM ローカルサーバーから Embedding ベクトルを取得する非同期クライアント。

    vLLM の OpenAI 互換 ``/v1/embeddings`` エンドポイントを使用する。
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._model = model
        self._client = httpx.AsyncClient(base_url=base_url)

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
        vectors = await self._request_embeddings([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストの Embedding ベクトルをバッチ取得する。

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
        return await self._request_embeddings(texts)

    async def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        """vLLM サーバーに Embedding リクエストを送信する。"""
        payload = {"model": self._model, "input": texts}
        try:
            response = await self._client.post("/embeddings", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"HTTP error: {e}") from e
        except httpx.ConnectError as e:
            raise EmbeddingError(f"Connection error: {e}") from e

        data = response.json()
        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]

    async def close(self) -> None:
        """内部 HTTP クライアントを閉じる。"""
        await self._client.aclose()

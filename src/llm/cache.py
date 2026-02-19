"""セマンティックキャッシュ。

クエリの Embedding をコサイン類似度で比較し、意味的に同等のクエリに対して
Redis キャッシュからレスポンスを返す。LLM API コールの削減が目的。
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import TYPE_CHECKING

import structlog
from redis.asyncio import Redis

from src.llm.providers.base import ChatMessage, LLMResponse, Role

if TYPE_CHECKING:
    from src.rag.embedder import EmbedderProtocol

logger = structlog.get_logger()

_CACHE_KEY_PREFIX = "llm_cache:"


class SemanticCache:
    """Embedding コサイン類似度ベースのセマンティックキャッシュ。

    Redis に ``llm_cache:<uuid>`` キーで embedding + レスポンスを格納し、
    SCAN で全エントリを線形走査して最良マッチを返す。

    障害時はメイン処理を止めず、warning ログを出力して None / 何もしないで返す。
    """

    def __init__(
        self,
        redis_client: Redis,
        embedder: EmbedderProtocol,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.95,
    ) -> None:
        if ttl_seconds <= 0:
            msg = "ttl_seconds must be positive"
            raise ValueError(msg)
        if not (0.0 <= similarity_threshold <= 1.0):
            msg = "similarity_threshold must be between 0.0 and 1.0"
            raise ValueError(msg)

        self._redis_client = redis_client
        self._embedder = embedder
        self._ttl_seconds = ttl_seconds
        self._similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    async def get(self, messages: list[ChatMessage]) -> LLMResponse | None:
        """キャッシュからセマンティック検索で LLMResponse を取得する。

        Args:
            messages: チャットメッセージリスト。

        Returns:
            類似度が閾値以上の最良マッチの LLMResponse。ミス時は None。
        """
        try:
            return await self._get_impl(messages)
        except Exception:
            logger.warning("semantic_cache_get_failed", exc_info=True)
            return None

    async def put(self, messages: list[ChatMessage], response: LLMResponse) -> None:
        """レスポンスをキャッシュに格納する。

        Args:
            messages: チャットメッセージリスト。
            response: 格納する LLMResponse。
        """
        try:
            await self._put_impl(messages, response)
        except Exception:
            logger.warning("semantic_cache_put_failed", exc_info=True)

    async def close(self) -> None:
        """Redis クライアントを閉じる。"""
        try:
            await self._redis_client.aclose()
        except Exception:
            logger.warning("semantic_cache_close_failed", exc_info=True)

    async def clear(self) -> None:
        """全キャッシュエントリを削除する。"""
        try:
            keys: list[bytes] = []
            cursor: int = 0
            while True:
                cursor, batch = await self._redis_client.scan(
                    cursor=cursor, match=f"{_CACHE_KEY_PREFIX}*", count=100
                )
                keys.extend(batch)
                if cursor == 0:
                    break
            if keys:
                await self._redis_client.delete(*keys)
        except Exception:
            logger.warning("semantic_cache_clear_failed", exc_info=True)

    # ------------------------------------------------------------------
    # static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """2つのベクトルのコサイン類似度を計算する。

        Args:
            vec_a: ベクトル A。
            vec_b: ベクトル B。

        Returns:
            コサイン類似度 (-1.0 〜 1.0)。ゼロベクトルの場合は 0.0。
        """
        dot = math.fsum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(math.fsum(a * a for a in vec_a))
        norm_b = math.sqrt(math.fsum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _extract_user_query(messages: list[ChatMessage]) -> str | None:
        """メッセージリストから最後のユーザーメッセージの content を抽出する。

        Args:
            messages: チャットメッセージリスト。

        Returns:
            最後のユーザーメッセージの content。見つからない場合は None。
        """
        for msg in reversed(messages):
            if msg.role == Role.user:
                return msg.content
        return None

    @staticmethod
    def _make_messages_hash(messages: list[ChatMessage]) -> str:
        """メッセージリストの SHA256 ハッシュを生成する。

        Args:
            messages: チャットメッセージリスト。

        Returns:
            64文字の16進ハッシュ文字列。
        """
        payload = json.dumps(
            [{"role": m.role.value, "content": m.content} for m in messages],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    async def _get_impl(self, messages: list[ChatMessage]) -> LLMResponse | None:
        """get() の内部実装。"""
        user_query = self._extract_user_query(messages)
        if user_query is None:
            return None

        query_embedding = await self._embedder.embed(user_query)

        best_similarity = -1.0
        best_response: LLMResponse | None = None

        cursor: int = 0
        while True:
            cursor, keys = await self._redis_client.scan(
                cursor=cursor, match=f"{_CACHE_KEY_PREFIX}*", count=100
            )
            for key in keys:
                raw = await self._redis_client.get(key)
                if raw is None:
                    continue
                entry = json.loads(raw)
                cached_embedding: list[float] = entry["embedding"]
                similarity = self._compute_similarity(query_embedding, cached_embedding)
                if similarity >= self._similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_response = LLMResponse(**entry["response"])
            if cursor == 0:
                break

        return best_response

    async def _put_impl(self, messages: list[ChatMessage], response: LLMResponse) -> None:
        """put() の内部実装。"""
        user_query = self._extract_user_query(messages)
        if user_query is None:
            return

        embedding = await self._embedder.embed(user_query)
        messages_hash = self._make_messages_hash(messages)

        entry = json.dumps(
            {
                "embedding": embedding,
                "response": response.model_dump(),
                "messages_hash": messages_hash,
            }
        )

        key = f"{_CACHE_KEY_PREFIX}{uuid.uuid4()}"
        await self._redis_client.set(key, entry, ex=self._ttl_seconds)

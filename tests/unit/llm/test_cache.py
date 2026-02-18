"""SemanticCache のユニットテスト。

キャッシュヒット/ミスの判定、TTLによる期限切れ、Redis障害時のグレースフル劣化を検証する。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from src.llm import CacheError
from src.llm.cache import SemanticCache
from src.llm.providers.base import ChatMessage, LLMResponse, Role, TokenUsage


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------
def _make_messages(user_content: str) -> list[ChatMessage]:
    """テスト用の ChatMessage リストを生成する。"""
    return [ChatMessage(role=Role.user, content=user_content)]


def _make_embedding(dims: int = 1024, value: float = 0.1) -> list[float]:
    """テスト用の embedding ベクトルを生成する。"""
    return [value] * dims


def _make_llm_response(content: str) -> LLMResponse:
    """テスト用の LLMResponse を生成する。"""
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        finish_reason="stop",
    )


def _make_cache_entry(embedding: list[float], response: LLMResponse, messages_hash: str) -> str:
    """Redis格納用のJSON文字列を生成する。"""
    return json.dumps(
        {
            "embedding": embedding,
            "response": response.model_dump(),
            "messages_hash": messages_hash,
        }
    )


# ---------------------------------------------------------------------------
# TestCacheError
# ---------------------------------------------------------------------------
class TestCacheError:
    """CacheError 例外クラスの基本動作を検証する。"""

    def test_cache_error_is_exception(self) -> None:
        """CacheError が Exception を継承していること。"""
        assert issubclass(CacheError, Exception)

    def test_cache_error_can_be_raised_with_message(self) -> None:
        """CacheError がメッセージ付きで raise できること。"""
        with pytest.raises(CacheError, match="test error"):
            raise CacheError("test error")


# ---------------------------------------------------------------------------
# TestSemanticCacheInit
# ---------------------------------------------------------------------------
class TestSemanticCacheInit:
    """SemanticCache コンストラクタを検証する。"""

    def test_creates_with_required_args(self) -> None:
        """redis_client と embedder のみで生成できること。"""
        redis = AsyncMock()
        embedder = AsyncMock()
        cache = SemanticCache(redis_client=redis, embedder=embedder)

        assert cache._redis_client is redis
        assert cache._embedder is embedder

    def test_default_ttl_seconds(self) -> None:
        """デフォルトの TTL が 3600 であること。"""
        cache = SemanticCache(redis_client=AsyncMock(), embedder=AsyncMock())

        assert cache._ttl_seconds == 3600

    def test_default_similarity_threshold(self) -> None:
        """デフォルトの類似度閾値が 0.95 であること。"""
        cache = SemanticCache(redis_client=AsyncMock(), embedder=AsyncMock())

        assert cache._similarity_threshold == 0.95

    def test_custom_ttl_seconds(self) -> None:
        """カスタム TTL が設定できること。"""
        cache = SemanticCache(redis_client=AsyncMock(), embedder=AsyncMock(), ttl_seconds=7200)

        assert cache._ttl_seconds == 7200

    def test_custom_similarity_threshold(self) -> None:
        """カスタム類似度閾値が設定できること。"""
        cache = SemanticCache(
            redis_client=AsyncMock(), embedder=AsyncMock(), similarity_threshold=0.90
        )

        assert cache._similarity_threshold == 0.90

    def test_ttl_must_be_positive(self) -> None:
        """TTL が 0 以下の場合に ValueError が発生すること。"""
        with pytest.raises(ValueError, match="ttl_seconds"):
            SemanticCache(redis_client=AsyncMock(), embedder=AsyncMock(), ttl_seconds=0)

    def test_threshold_must_be_between_0_and_1(self) -> None:
        """閾値が 0-1 の範囲外の場合に ValueError が発生すること。"""
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticCache(redis_client=AsyncMock(), embedder=AsyncMock(), similarity_threshold=1.5)

    def test_threshold_must_not_be_negative(self) -> None:
        """閾値が負の場合に ValueError が発生すること。"""
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticCache(redis_client=AsyncMock(), embedder=AsyncMock(), similarity_threshold=-0.1)


# ---------------------------------------------------------------------------
# TestComputeSimilarity
# ---------------------------------------------------------------------------
class TestComputeSimilarity:
    """_compute_similarity 静的メソッド（コサイン類似度）を検証する。"""

    def test_identical_vectors_return_1(self) -> None:
        """同一ベクトルの類似度が 1.0 であること。"""
        vec = [1.0, 2.0, 3.0]

        result = SemanticCache._compute_similarity(vec, vec)

        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors_return_0(self) -> None:
        """直交ベクトルの類似度が 0.0 であること。"""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]

        result = SemanticCache._compute_similarity(vec_a, vec_b)

        assert result == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_1(self) -> None:
        """逆向きベクトルの類似度が -1.0 であること。"""
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]

        result = SemanticCache._compute_similarity(vec_a, vec_b)

        assert result == pytest.approx(-1.0)

    def test_zero_vector_a_returns_0(self) -> None:
        """ゼロベクトル (a) の場合に 0.0 を返すこと。"""
        vec_a = [0.0, 0.0]
        vec_b = [1.0, 1.0]

        result = SemanticCache._compute_similarity(vec_a, vec_b)

        assert result == pytest.approx(0.0)

    def test_zero_vector_b_returns_0(self) -> None:
        """ゼロベクトル (b) の場合に 0.0 を返すこと。"""
        vec_a = [1.0, 1.0]
        vec_b = [0.0, 0.0]

        result = SemanticCache._compute_similarity(vec_a, vec_b)

        assert result == pytest.approx(0.0)

    def test_similar_vectors_return_high_similarity(self) -> None:
        """類似ベクトルが高い類似度を返すこと。"""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.1, 2.1, 3.1]

        result = SemanticCache._compute_similarity(vec_a, vec_b)

        assert result > 0.99

    def test_dissimilar_vectors_return_low_similarity(self) -> None:
        """非類似ベクトルが低い類似度を返すこと。"""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 0.0, 1.0]

        result = SemanticCache._compute_similarity(vec_a, vec_b)

        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestExtractUserQuery
# ---------------------------------------------------------------------------
class TestExtractUserQuery:
    """_extract_user_query 静的メソッドを検証する。"""

    def test_extracts_user_message_content(self) -> None:
        """ユーザーメッセージの content を抽出すること。"""
        messages = [ChatMessage(role=Role.user, content="こんにちは")]

        result = SemanticCache._extract_user_query(messages)

        assert result == "こんにちは"

    def test_returns_none_for_empty_messages(self) -> None:
        """空のメッセージリストで None を返すこと。"""
        result = SemanticCache._extract_user_query([])

        assert result is None

    def test_returns_none_when_no_user_message(self) -> None:
        """ユーザーメッセージがない場合に None を返すこと。"""
        messages = [ChatMessage(role=Role.system, content="You are a helpful assistant.")]

        result = SemanticCache._extract_user_query(messages)

        assert result is None

    def test_extracts_last_user_message_in_multi_turn(self) -> None:
        """マルチターンで最後のユーザーメッセージを抽出すること。"""
        messages = [
            ChatMessage(role=Role.user, content="最初の質問"),
            ChatMessage(role=Role.assistant, content="回答1"),
            ChatMessage(role=Role.user, content="最後の質問"),
        ]

        result = SemanticCache._extract_user_query(messages)

        assert result == "最後の質問"


# ---------------------------------------------------------------------------
# TestSemanticCacheGet
# ---------------------------------------------------------------------------
class TestSemanticCacheGet:
    """SemanticCache.get() を検証する。"""

    @pytest.fixture
    def embedder(self) -> AsyncMock:
        """テスト用の embedder モック。"""
        mock = AsyncMock()
        mock.embed.return_value = _make_embedding()
        return mock

    @pytest.fixture
    def redis_client(self) -> AsyncMock:
        """テスト用の Redis クライアントモック。"""
        return AsyncMock()

    async def test_cache_hit_returns_llm_response(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """キャッシュヒット時に LLMResponse を返すこと。"""
        cached_embedding = _make_embedding()
        cached_response = _make_llm_response("cached answer")
        entry = _make_cache_entry(cached_embedding, cached_response, "hash123")

        redis_client.scan.return_value = (0, [b"llm_cache:abc"])
        redis_client.get.return_value = entry

        cache = SemanticCache(
            redis_client=redis_client, embedder=embedder, similarity_threshold=0.90
        )
        result = await cache.get(_make_messages("test query"))

        assert result is not None
        assert result.content == "cached answer"
        assert result.model == "test-model"

    async def test_cache_miss_returns_none(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """キャッシュが空の場合に None を返すこと。"""
        redis_client.scan.return_value = (0, [])

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        result = await cache.get(_make_messages("test query"))

        assert result is None

    async def test_below_threshold_returns_none(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """類似度が閾値未満の場合に None を返すこと。"""
        # 直交ベクトルを格納 -> 類似度 0.0
        cached_embedding = [0.0] * 1023 + [1.0]
        query_embedding = [1.0] + [0.0] * 1023
        embedder.embed.return_value = query_embedding

        cached_response = _make_llm_response("cached answer")
        entry = _make_cache_entry(cached_embedding, cached_response, "hash123")

        redis_client.scan.return_value = (0, [b"llm_cache:abc"])
        redis_client.get.return_value = entry

        cache = SemanticCache(
            redis_client=redis_client, embedder=embedder, similarity_threshold=0.95
        )
        result = await cache.get(_make_messages("test query"))

        assert result is None

    async def test_returns_best_match_above_threshold(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """閾値以上の最良マッチを返すこと。"""
        # クエリ embedding
        query_emb = [1.0] * 1024
        embedder.embed.return_value = query_emb

        # エントリ1: 完全一致
        entry1_emb = [1.0] * 1024
        entry1_resp = _make_llm_response("best match")
        entry1 = _make_cache_entry(entry1_emb, entry1_resp, "hash1")

        # エントリ2: やや異なる
        entry2_emb = [0.9] * 1024
        entry2_resp = _make_llm_response("second match")
        entry2 = _make_cache_entry(entry2_emb, entry2_resp, "hash2")

        redis_client.scan.return_value = (0, [b"llm_cache:a", b"llm_cache:b"])
        redis_client.get.side_effect = [entry1, entry2]

        cache = SemanticCache(
            redis_client=redis_client, embedder=embedder, similarity_threshold=0.90
        )
        result = await cache.get(_make_messages("test"))

        assert result is not None
        assert result.content == "best match"

    async def test_redis_error_returns_none(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """Redis 障害時に None を返すこと（メイン処理を止めない）。"""
        from redis.exceptions import RedisError

        redis_client.scan.side_effect = RedisError("connection refused")

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        result = await cache.get(_make_messages("test"))

        assert result is None

    async def test_embedding_error_returns_none(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """Embedding 障害時に None を返すこと。"""
        from src.rag.embedder import EmbeddingError

        embedder.embed.side_effect = EmbeddingError("server down")

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        result = await cache.get(_make_messages("test"))

        assert result is None

    async def test_no_user_message_returns_none(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """ユーザーメッセージがない場合に None を返すこと。"""
        messages = [ChatMessage(role=Role.system, content="system prompt")]

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        result = await cache.get(messages)

        assert result is None

    async def test_scan_pagination(self, redis_client: AsyncMock, embedder: AsyncMock) -> None:
        """SCAN のカーソルベースページネーションが正しく動作すること。"""
        cached_embedding = _make_embedding()
        cached_response = _make_llm_response("found in page 2")
        entry = _make_cache_entry(cached_embedding, cached_response, "hash1")

        # 1回目: cursor=1 で続きがある, 2回目: cursor=0 で終了
        redis_client.scan.side_effect = [
            (1, []),
            (0, [b"llm_cache:abc"]),
        ]
        redis_client.get.return_value = entry

        cache = SemanticCache(
            redis_client=redis_client, embedder=embedder, similarity_threshold=0.90
        )
        result = await cache.get(_make_messages("test"))

        assert result is not None
        assert result.content == "found in page 2"
        assert redis_client.scan.call_count == 2

    async def test_response_preserves_usage_and_finish_reason(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """キャッシュヒット時に usage と finish_reason が保持されること。"""
        cached_response = _make_llm_response("cached")
        entry = _make_cache_entry(_make_embedding(), cached_response, "hash1")

        redis_client.scan.return_value = (0, [b"llm_cache:abc"])
        redis_client.get.return_value = entry

        cache = SemanticCache(
            redis_client=redis_client, embedder=embedder, similarity_threshold=0.90
        )
        result = await cache.get(_make_messages("test"))

        assert result is not None
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.finish_reason == "stop"


# ---------------------------------------------------------------------------
# TestSemanticCachePut
# ---------------------------------------------------------------------------
class TestSemanticCachePut:
    """SemanticCache.put() を検証する。"""

    @pytest.fixture
    def embedder(self) -> AsyncMock:
        mock = AsyncMock()
        mock.embed.return_value = _make_embedding()
        return mock

    @pytest.fixture
    def redis_client(self) -> AsyncMock:
        return AsyncMock()

    async def test_stores_entry_in_redis(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """エントリが Redis に格納されること。"""
        cache = SemanticCache(redis_client=redis_client, embedder=embedder)

        await cache.put(_make_messages("test query"), _make_llm_response("answer"))

        redis_client.set.assert_called_once()

    async def test_key_has_llm_cache_prefix(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """キーが llm_cache: プレフィックスを持つこと。"""
        cache = SemanticCache(redis_client=redis_client, embedder=embedder)

        await cache.put(_make_messages("test"), _make_llm_response("answer"))

        call_args = redis_client.set.call_args
        key = call_args[0][0] if call_args[0] else call_args[1].get("name", "")
        assert key.startswith("llm_cache:")

    async def test_ttl_is_set(self, redis_client: AsyncMock, embedder: AsyncMock) -> None:
        """TTL が設定されること。"""
        cache = SemanticCache(redis_client=redis_client, embedder=embedder, ttl_seconds=1800)

        await cache.put(_make_messages("test"), _make_llm_response("answer"))

        call_kwargs = redis_client.set.call_args[1]
        assert call_kwargs.get("ex") == 1800

    async def test_stored_value_is_valid_json(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """格納値が有効な JSON であること。"""
        cache = SemanticCache(redis_client=redis_client, embedder=embedder)

        await cache.put(_make_messages("test"), _make_llm_response("answer"))

        call_args = redis_client.set.call_args
        value = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("value", "")
        data = json.loads(value)
        assert "embedding" in data
        assert "response" in data
        assert "messages_hash" in data

    async def test_stored_embedding_matches(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """格納された embedding がクエリの embedding と一致すること。"""
        expected_emb = _make_embedding(dims=1024, value=0.5)
        embedder.embed.return_value = expected_emb

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        await cache.put(_make_messages("test"), _make_llm_response("answer"))

        call_args = redis_client.set.call_args
        value = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("value", "")
        data = json.loads(value)
        assert data["embedding"] == expected_emb

    async def test_stored_response_can_be_deserialized(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """格納された response が LLMResponse にデシリアライズできること。"""
        original = _make_llm_response("answer text")
        cache = SemanticCache(redis_client=redis_client, embedder=embedder)

        await cache.put(_make_messages("test"), original)

        call_args = redis_client.set.call_args
        value = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("value", "")
        data = json.loads(value)
        restored = LLMResponse(**data["response"])
        assert restored.content == "answer text"
        assert restored.model == "test-model"

    async def test_redis_error_does_not_raise(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """Redis 障害時に例外を raise しないこと。"""
        from redis.exceptions import RedisError

        redis_client.set.side_effect = RedisError("write failed")

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        # 例外が発生しないことを確認
        await cache.put(_make_messages("test"), _make_llm_response("answer"))

    async def test_embedding_error_does_not_raise(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """Embedding 障害時に例外を raise しないこと。"""
        from src.rag.embedder import EmbeddingError

        embedder.embed.side_effect = EmbeddingError("server down")

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        await cache.put(_make_messages("test"), _make_llm_response("answer"))

        redis_client.set.assert_not_called()

    async def test_no_user_message_does_not_store(
        self, redis_client: AsyncMock, embedder: AsyncMock
    ) -> None:
        """ユーザーメッセージがない場合に格納しないこと。"""
        messages = [ChatMessage(role=Role.system, content="system")]

        cache = SemanticCache(redis_client=redis_client, embedder=embedder)
        await cache.put(messages, _make_llm_response("answer"))

        redis_client.set.assert_not_called()


# ---------------------------------------------------------------------------
# TestSemanticCacheClose
# ---------------------------------------------------------------------------
class TestSemanticCacheClose:
    """SemanticCache.close() を検証する。"""

    async def test_closes_redis_client(self) -> None:
        """Redis クライアントの aclose が呼ばれること。"""
        redis_client = AsyncMock()
        cache = SemanticCache(redis_client=redis_client, embedder=AsyncMock())

        await cache.close()

        redis_client.aclose.assert_called_once()

    async def test_close_error_does_not_raise(self) -> None:
        """close 時のエラーが例外を raise しないこと。"""
        redis_client = AsyncMock()
        redis_client.aclose.side_effect = Exception("close failed")

        cache = SemanticCache(redis_client=redis_client, embedder=AsyncMock())
        await cache.close()


# ---------------------------------------------------------------------------
# TestSemanticCacheClear
# ---------------------------------------------------------------------------
class TestSemanticCacheClear:
    """SemanticCache.clear() を検証する。"""

    async def test_deletes_all_cache_keys(self) -> None:
        """全キャッシュキーが削除されること。"""
        redis_client = AsyncMock()
        redis_client.scan.return_value = (0, [b"llm_cache:a", b"llm_cache:b"])

        cache = SemanticCache(redis_client=redis_client, embedder=AsyncMock())
        await cache.clear()

        redis_client.delete.assert_called_once_with(b"llm_cache:a", b"llm_cache:b")

    async def test_clear_empty_cache(self) -> None:
        """空のキャッシュで clear が正常動作すること。"""
        redis_client = AsyncMock()
        redis_client.scan.return_value = (0, [])

        cache = SemanticCache(redis_client=redis_client, embedder=AsyncMock())
        await cache.clear()

        redis_client.delete.assert_not_called()

    async def test_clear_redis_error_does_not_raise(self) -> None:
        """Redis 障害時に例外を raise しないこと。"""
        from redis.exceptions import RedisError

        redis_client = AsyncMock()
        redis_client.scan.side_effect = RedisError("connection refused")

        cache = SemanticCache(redis_client=redis_client, embedder=AsyncMock())
        await cache.clear()


# ---------------------------------------------------------------------------
# TestMakeMessagesHash
# ---------------------------------------------------------------------------
class TestMakeMessagesHash:
    """_make_messages_hash 静的メソッドを検証する。"""

    def test_deterministic_hash(self) -> None:
        """同じメッセージに対して同じハッシュを返すこと。"""
        messages = _make_messages("test query")

        hash1 = SemanticCache._make_messages_hash(messages)
        hash2 = SemanticCache._make_messages_hash(messages)

        assert hash1 == hash2

    def test_different_messages_produce_different_hashes(self) -> None:
        """異なるメッセージに対して異なるハッシュを返すこと。"""
        hash1 = SemanticCache._make_messages_hash(_make_messages("query A"))
        hash2 = SemanticCache._make_messages_hash(_make_messages("query B"))

        assert hash1 != hash2

    def test_hash_is_hex_string(self) -> None:
        """ハッシュが16進文字列であること。"""
        result = SemanticCache._make_messages_hash(_make_messages("test"))

        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest
        int(result, 16)  # 16進数としてパースできること

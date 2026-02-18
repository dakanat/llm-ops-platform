"""レート制限ミドルウェアのユニットテスト。

Redis Token Bucket アルゴリズムによるレート制限の検証:
初期化、クライアントIP抽出、非HTTPパススルー、無効モード、
制限内リクエスト、制限超過、トークン補充、障害時の graceful degradation。
"""

import json
import math
from collections.abc import MutableMapping
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from src.api.middleware.rate_limit import RateLimitMiddleware
from starlette.types import Receive, Scope, Send

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lua_result(allowed: int, remaining: int, retry_after: float = 0.0) -> list[int | bytes]:
    """Simulate the return value of the Redis Lua script.

    Returns [allowed, remaining_bytes, retry_after_bytes].
    """
    return [allowed, str(remaining).encode(), str(retry_after).encode()]


def _build_scope(
    *,
    scope_type: str = "http",
    path: str = "/test",
    method: str = "GET",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("127.0.0.1", 12345),
) -> dict[str, Any]:
    """Build an ASGI scope dict for testing."""
    scope: dict[str, Any] = {"type": scope_type, "path": path, "method": method}
    if headers is not None:
        scope["headers"] = headers
    else:
        scope["headers"] = []
    if client is not None:
        scope["client"] = client
    return scope


def _make_app(
    *,
    redis_client: Any = None,
    enabled: bool = True,
    requests_per_minute: int = 60,
    burst_size: int = 10,
) -> tuple[RateLimitMiddleware, AsyncMock]:
    """Create a RateLimitMiddleware wrapping a mock inner app."""
    inner_app = AsyncMock()

    async def fake_app(scope: Scope, receive: Receive, send: Send) -> None:
        # Simulate http.response.start + http.response.body
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"]],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"status":"ok"}',
            }
        )

    inner_app.side_effect = fake_app

    middleware = RateLimitMiddleware(
        inner_app,
        redis_client=redis_client,
        enabled=enabled,
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
    )
    return middleware, inner_app


# ===========================================================================
# TestRateLimitMiddlewareInit
# ===========================================================================


class TestRateLimitMiddlewareInit:
    """ミドルウェア初期化パラメータの検証。"""

    def test_defaults_stored(self) -> None:
        """デフォルトパラメータが正しく保存されること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app)
        assert mw.app is mock_app
        assert mw._enabled is True
        assert mw._requests_per_minute == 60
        assert mw._burst_size == 10

    def test_disabled_mode(self) -> None:
        """enabled=False が保存されること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app, enabled=False)
        assert mw._enabled is False

    def test_no_redis_client(self) -> None:
        """redis_client=None が許容されること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app, redis_client=None)
        assert mw._redis_client is None

    def test_refill_rate_calculation(self) -> None:
        """リフィルレートが requests_per_minute / 60 で計算されること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app, requests_per_minute=120)
        assert mw._refill_rate == pytest.approx(2.0)

    def test_custom_params(self) -> None:
        """カスタムパラメータが保存されること。"""
        mock_redis = AsyncMock()
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(
            mock_app,
            redis_client=mock_redis,
            enabled=True,
            requests_per_minute=30,
            burst_size=5,
        )
        assert mw._redis_client is mock_redis
        assert mw._requests_per_minute == 30
        assert mw._burst_size == 5
        assert mw._refill_rate == pytest.approx(0.5)


# ===========================================================================
# TestClientIpExtraction
# ===========================================================================


class TestClientIpExtraction:
    """クライアントIP抽出ロジックの検証。"""

    def test_from_scope_client(self) -> None:
        """scope['client'] からIPを取得すること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app)
        scope = _build_scope(client=("192.168.1.1", 8080))
        assert mw._get_client_ip(scope) == "192.168.1.1"

    def test_from_x_forwarded_for(self) -> None:
        """X-Forwarded-For ヘッダーからIPを取得すること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app)
        scope = _build_scope(
            headers=[(b"x-forwarded-for", b"10.0.0.1, 10.0.0.2")],
            client=("127.0.0.1", 12345),
        )
        assert mw._get_client_ip(scope) == "10.0.0.1"

    def test_xff_takes_precedence_over_scope_client(self) -> None:
        """X-Forwarded-For が scope['client'] より優先されること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app)
        scope = _build_scope(
            headers=[(b"x-forwarded-for", b"203.0.113.50")],
            client=("127.0.0.1", 12345),
        )
        assert mw._get_client_ip(scope) == "203.0.113.50"

    def test_unknown_fallback(self) -> None:
        """client情報もXFFもない場合に 'unknown' を返すこと。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app)
        scope: dict[str, Any] = {"type": "http", "headers": []}
        assert mw._get_client_ip(scope) == "unknown"

    def test_single_xff_ip(self) -> None:
        """X-Forwarded-For にIPが1つだけの場合も正しく取得すること。"""
        mock_app = AsyncMock()
        mw = RateLimitMiddleware(mock_app)
        scope = _build_scope(
            headers=[(b"x-forwarded-for", b"10.0.0.99")],
            client=("127.0.0.1", 12345),
        )
        assert mw._get_client_ip(scope) == "10.0.0.99"


# ===========================================================================
# TestNonHttpPassthrough
# ===========================================================================


class TestNonHttpPassthrough:
    """HTTP以外のスコープはそのまま通過する。"""

    async def test_lifespan_scope_passes_through(self) -> None:
        """lifespan スコープはミドルウェアをバイパスすること。"""
        inner_called = False

        async def mock_app(scope: Scope, receive: Receive, send: Send) -> None:
            nonlocal inner_called
            inner_called = True

        mw = RateLimitMiddleware(mock_app)
        await mw({"type": "lifespan"}, AsyncMock(), AsyncMock())
        assert inner_called

    async def test_websocket_scope_passes_through(self) -> None:
        """websocket スコープはミドルウェアをバイパスすること。"""
        inner_called = False

        async def mock_app(scope: Scope, receive: Receive, send: Send) -> None:
            nonlocal inner_called
            inner_called = True

        mw = RateLimitMiddleware(mock_app)
        await mw({"type": "websocket"}, AsyncMock(), AsyncMock())
        assert inner_called


# ===========================================================================
# TestDisabledMode
# ===========================================================================


class TestDisabledMode:
    """無効時・Redis未接続時はリクエストをそのまま通過させる。"""

    async def test_disabled_passes_through(self) -> None:
        """enabled=False のときリクエストが通過すること。"""
        middleware, inner_app = _make_app(enabled=False)
        scope = _build_scope()
        send = AsyncMock()
        await middleware(scope, AsyncMock(), send)
        inner_app.assert_called_once()

    async def test_no_redis_client_passes_through(self) -> None:
        """redis_client=None のときリクエストが通過すること。"""
        middleware, inner_app = _make_app(redis_client=None, enabled=True)
        scope = _build_scope()
        send = AsyncMock()
        await middleware(scope, AsyncMock(), send)
        inner_app.assert_called_once()


# ===========================================================================
# TestRequestsWithinLimit
# ===========================================================================


class TestRequestsWithinLimit:
    """制限内リクエストの検証。"""

    async def test_first_request_allowed(self) -> None:
        """最初のリクエストが許可されること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = _make_lua_result(1, 9)
        middleware, inner_app = _make_app(redis_client=mock_redis, burst_size=10)

        scope = _build_scope()
        send = AsyncMock()
        await middleware(scope, AsyncMock(), send)
        inner_app.assert_called_once()

    async def test_rate_limit_headers_present(self) -> None:
        """成功レスポンスにレート制限ヘッダーが含まれること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = _make_lua_result(1, 9)
        middleware, _ = _make_app(redis_client=mock_redis, burst_size=10)

        scope = _build_scope()
        sent_messages: list[MutableMapping[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(message)

        await middleware(scope, AsyncMock(), capture_send)

        start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = {k: v for k, v in start_msg["headers"]}
        assert b"x-ratelimit-limit" in header_dict
        assert b"x-ratelimit-remaining" in header_dict
        assert b"x-ratelimit-reset" in header_dict

    async def test_multiple_requests_within_burst(self) -> None:
        """バースト内の複数リクエストがすべて許可されること。"""
        mock_redis = AsyncMock()
        # Each call returns decreasing remaining
        mock_redis.eval.side_effect = [
            _make_lua_result(1, 9),
            _make_lua_result(1, 8),
            _make_lua_result(1, 7),
            _make_lua_result(1, 6),
            _make_lua_result(1, 5),
        ]
        middleware, inner_app = _make_app(redis_client=mock_redis, burst_size=10)

        for _ in range(5):
            scope = _build_scope()
            await middleware(scope, AsyncMock(), AsyncMock())

        assert inner_app.call_count == 5

    async def test_remaining_decreases(self) -> None:
        """リクエストごとに remaining が減少すること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.side_effect = [
            _make_lua_result(1, 9),
            _make_lua_result(1, 8),
        ]
        middleware, _ = _make_app(redis_client=mock_redis, burst_size=10)

        async def _do_request(mw: RateLimitMiddleware) -> bytes:
            scope = _build_scope()
            messages: list[MutableMapping[str, Any]] = []

            async def send(message: MutableMapping[str, Any]) -> None:
                messages.append(message)

            await mw(scope, AsyncMock(), send)
            start_msg = next(m for m in messages if m["type"] == "http.response.start")
            header_dict = {k: v for k, v in start_msg["headers"]}
            result: bytes = header_dict[b"x-ratelimit-remaining"]
            return result

        remaining_1 = await _do_request(middleware)
        remaining_2 = await _do_request(middleware)
        assert remaining_1 == b"9"
        assert remaining_2 == b"8"


# ===========================================================================
# TestRequestsExceedingLimit
# ===========================================================================


class TestRequestsExceedingLimit:
    """制限超過リクエストの検証。"""

    async def test_429_when_bucket_empty(self) -> None:
        """バケットが空のとき429が返ること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = _make_lua_result(0, 0, 1.5)
        middleware, inner_app = _make_app(redis_client=mock_redis)

        scope = _build_scope()
        sent_messages: list[MutableMapping[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(message)

        await middleware(scope, AsyncMock(), capture_send)

        start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
        assert start_msg["status"] == 429
        inner_app.assert_not_called()

    async def test_429_json_body(self) -> None:
        """429レスポンスのボディが正しいJSON形式であること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = _make_lua_result(0, 0, 1.5)
        middleware, _ = _make_app(redis_client=mock_redis)

        scope = _build_scope()
        sent_messages: list[MutableMapping[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(message)

        await middleware(scope, AsyncMock(), capture_send)

        body_msg = next(m for m in sent_messages if m["type"] == "http.response.body")
        body = json.loads(body_msg["body"])
        assert body == {"detail": "Rate limit exceeded. Please retry later."}

    async def test_429_retry_after_header(self) -> None:
        """429レスポンスに Retry-After ヘッダーが含まれること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = _make_lua_result(0, 0, 1.5)
        middleware, _ = _make_app(redis_client=mock_redis)

        scope = _build_scope()
        sent_messages: list[MutableMapping[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(message)

        await middleware(scope, AsyncMock(), capture_send)

        start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = {k: v for k, v in start_msg["headers"]}
        assert b"retry-after" in header_dict
        assert int(header_dict[b"retry-after"]) == math.ceil(1.5)

    async def test_429_content_type(self) -> None:
        """429レスポンスの Content-Type が application/json であること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = _make_lua_result(0, 0, 1.0)
        middleware, _ = _make_app(redis_client=mock_redis)

        scope = _build_scope()
        sent_messages: list[MutableMapping[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(message)

        await middleware(scope, AsyncMock(), capture_send)

        start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = {k: v for k, v in start_msg["headers"]}
        assert header_dict[b"content-type"] == b"application/json"

    async def test_rate_limit_headers_on_429(self) -> None:
        """429レスポンスにもレート制限ヘッダーが含まれること。"""
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = _make_lua_result(0, 0, 2.0)
        middleware, _ = _make_app(redis_client=mock_redis, burst_size=10)

        scope = _build_scope()
        sent_messages: list[MutableMapping[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(message)

        await middleware(scope, AsyncMock(), capture_send)

        start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = {k: v for k, v in start_msg["headers"]}
        assert b"x-ratelimit-limit" in header_dict
        assert b"x-ratelimit-remaining" in header_dict
        assert b"x-ratelimit-reset" in header_dict


# ===========================================================================
# TestTokenRefill
# ===========================================================================


class TestTokenRefill:
    """トークン補充の検証。"""

    async def test_tokens_refill_over_time(self) -> None:
        """時間経過でトークンが補充されること。"""
        mock_redis = AsyncMock()
        # First call: bucket exhausted
        # Second call: after time passes, tokens refilled
        mock_redis.eval.side_effect = [
            _make_lua_result(0, 0, 1.0),
            _make_lua_result(1, 4),
        ]
        middleware, inner_app = _make_app(redis_client=mock_redis, burst_size=10)

        # First request: rejected
        scope = _build_scope()
        sent_messages_1: list[MutableMapping[str, Any]] = []

        async def capture_send_1(message: MutableMapping[str, Any]) -> None:
            sent_messages_1.append(message)

        await middleware(scope, AsyncMock(), capture_send_1)
        start_1 = next(m for m in sent_messages_1 if m["type"] == "http.response.start")
        assert start_1["status"] == 429

        # Second request: allowed (simulating time has passed via mock return)
        scope = _build_scope()
        sent_messages_2: list[MutableMapping[str, Any]] = []

        async def capture_send_2(message: MutableMapping[str, Any]) -> None:
            sent_messages_2.append(message)

        await middleware(scope, AsyncMock(), capture_send_2)
        inner_app.assert_called_once()

    async def test_tokens_capped_at_burst_size(self) -> None:
        """トークン数がバーストサイズを超えないこと。"""
        mock_redis = AsyncMock()
        # Lua script should cap remaining at burst_size
        mock_redis.eval.return_value = _make_lua_result(1, 10)
        middleware, _ = _make_app(redis_client=mock_redis, burst_size=10)

        scope = _build_scope()
        sent_messages: list[MutableMapping[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(message)

        await middleware(scope, AsyncMock(), capture_send)

        start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = {k: v for k, v in start_msg["headers"]}
        remaining = int(header_dict[b"x-ratelimit-remaining"])
        assert remaining <= 10


# ===========================================================================
# TestGracefulDegradation
# ===========================================================================


class TestGracefulDegradation:
    """Redis障害時のgraceful degradation。"""

    async def test_connection_error_allows_request(self) -> None:
        """Redis ConnectionError 時にリクエストが通過すること。"""
        from redis.exceptions import ConnectionError as RedisConnectionError

        mock_redis = AsyncMock()
        mock_redis.eval.side_effect = RedisConnectionError("connection refused")
        middleware, inner_app = _make_app(redis_client=mock_redis)

        scope = _build_scope()
        await middleware(scope, AsyncMock(), AsyncMock())
        inner_app.assert_called_once()

    async def test_timeout_error_allows_request(self) -> None:
        """Redis TimeoutError 時にリクエストが通過すること。"""
        from redis.exceptions import TimeoutError as RedisTimeoutError

        mock_redis = AsyncMock()
        mock_redis.eval.side_effect = RedisTimeoutError("timeout")
        middleware, inner_app = _make_app(redis_client=mock_redis)

        scope = _build_scope()
        await middleware(scope, AsyncMock(), AsyncMock())
        inner_app.assert_called_once()

    async def test_warning_logged_on_redis_error(self) -> None:
        """Redis エラー時に warning ログが出力されること。"""
        from redis.exceptions import ConnectionError as RedisConnectionError

        mock_redis = AsyncMock()
        mock_redis.eval.side_effect = RedisConnectionError("connection refused")
        middleware, _ = _make_app(redis_client=mock_redis)

        scope = _build_scope()
        with patch("src.api.middleware.rate_limit.logger") as mock_logger:
            await middleware(scope, AsyncMock(), AsyncMock())
            mock_logger.warning.assert_called_once()

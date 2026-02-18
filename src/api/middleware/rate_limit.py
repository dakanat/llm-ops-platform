"""ASGI middleware for rate limiting using Redis Token Bucket algorithm.

Implements per-client IP rate limiting with a Token Bucket stored in Redis.
An atomic Lua script handles bucket state (refill + consume) to avoid
race conditions. On Redis failure the middleware degrades gracefully,
allowing requests through with a warning log.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) to
avoid breaking SSE streaming responses.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import MutableMapping
from typing import Any

import structlog
from starlette.types import ASGIApp, Receive, Scope, Send

logger = structlog.get_logger()

# Redis Lua script for atomic token bucket operation.
# KEYS[1] = rate_limit:{client_ip}
# ARGV[1] = burst_size (max tokens)
# ARGV[2] = refill_rate (tokens per second)
# ARGV[3] = now (current time in seconds as float string)
# ARGV[4] = ttl (key TTL in seconds)
#
# Returns: [allowed (0|1), remaining (string), retry_after (string)]
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local burst_size = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local tokens = burst_size
local last_refill = now

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
if data[1] ~= false then
    tokens = tonumber(data[1])
    last_refill = tonumber(data[2])
end

-- Refill tokens based on elapsed time
local elapsed = now - last_refill
local refill = elapsed * refill_rate
tokens = math.min(burst_size, tokens + refill)
last_refill = now

-- Try to consume one token
if tokens >= 1 then
    tokens = tokens - 1
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))
    redis.call('EXPIRE', key, ttl)
    return {1, tostring(math.floor(tokens)), "0"}
else
    -- Calculate time until next token
    local deficit = 1 - tokens
    local retry_after = deficit / refill_rate
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))
    redis.call('EXPIRE', key, ttl)
    return {0, "0", tostring(retry_after)}
end
"""

_KEY_PREFIX = "rate_limit:"
_KEY_TTL_SECONDS = 120


class RateLimitMiddleware:
    """Pure ASGI middleware for per-client rate limiting via Redis Token Bucket.

    Args:
        app: The inner ASGI application.
        redis_client: An async Redis client instance. ``None`` disables limiting.
        enabled: Whether rate limiting is active.
        requests_per_minute: Sustained request rate (tokens refilled per minute).
        burst_size: Maximum tokens in the bucket (peak burst capacity).
    """

    def __init__(
        self,
        app: ASGIApp,
        redis_client: Any | None = None,
        enabled: bool = True,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ) -> None:
        self.app = app
        self._redis_client = redis_client
        self._enabled = enabled
        self._requests_per_minute = requests_per_minute
        self._burst_size = burst_size
        self._refill_rate: float = requests_per_minute / 60.0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process an ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self._enabled or self._redis_client is None:
            await self.app(scope, receive, send)
            return

        client_ip = self._get_client_ip(scope)
        key = f"{_KEY_PREFIX}{client_ip}"

        try:
            result = await self._redis_client.eval(
                _TOKEN_BUCKET_LUA,
                1,
                key,
                str(self._burst_size),
                str(self._refill_rate),
                str(time.time()),
                str(_KEY_TTL_SECONDS),
            )
            allowed = int(result[0])
            remaining = result[1] if isinstance(result[1], bytes) else str(result[1]).encode()
            retry_after_raw = result[2] if isinstance(result[2], bytes) else str(result[2]).encode()
            remaining_str = remaining.decode() if isinstance(remaining, bytes) else str(remaining)
            retry_after_str = (
                retry_after_raw.decode()
                if isinstance(retry_after_raw, bytes)
                else str(retry_after_raw)
            )
            retry_after = float(retry_after_str)
        except Exception:
            logger.warning("rate_limit_redis_error", exc_info=True)
            await self.app(scope, receive, send)
            return

        if not allowed:
            await self._send_429(send, remaining_str, retry_after)
            return

        # Inject rate limit headers into successful responses
        rate_limit_headers = self._build_rate_limit_headers(remaining_str, retry_after)

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers: list[Any] = list(message.get("headers", []))
                headers.extend(rate_limit_headers)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)

    def _get_client_ip(self, scope: Scope) -> str:
        """Extract the client IP from the ASGI scope.

        Checks ``X-Forwarded-For`` header first (first IP), then falls back
        to ``scope["client"][0]``, and finally to ``"unknown"``.
        """
        headers = dict(scope.get("headers", []))
        xff = headers.get(b"x-forwarded-for")
        if xff is not None:
            xff_str = xff.decode("utf-8") if isinstance(xff, bytes) else xff
            return xff_str.split(",")[0].strip()

        client = scope.get("client")
        if client is not None:
            return str(client[0])

        return "unknown"

    def _build_rate_limit_headers(self, remaining: str, retry_after: float) -> list[list[bytes]]:
        """Build X-RateLimit-* response headers."""
        reset_seconds = math.ceil(retry_after) if retry_after > 0 else 0
        return [
            [b"x-ratelimit-limit", str(self._burst_size).encode()],
            [b"x-ratelimit-remaining", remaining.encode()],
            [b"x-ratelimit-reset", str(reset_seconds).encode()],
        ]

    async def _send_429(self, send: Send, remaining: str, retry_after: float) -> None:
        """Send a 429 Too Many Requests response."""
        retry_after_ceil = math.ceil(retry_after)
        body = json.dumps({"detail": "Rate limit exceeded. Please retry later."}).encode()

        headers: list[list[bytes]] = [
            [b"content-type", b"application/json"],
            [b"retry-after", str(retry_after_ceil).encode()],
            *self._build_rate_limit_headers(remaining, retry_after),
        ]

        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )

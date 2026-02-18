"""ASGI middleware for request logging with request ID propagation.

Assigns a unique request ID to each HTTP request (from ``X-Request-ID``
header or auto-generated UUID4), binds it to structlog contextvars, and
logs ``request_started`` / ``request_completed`` events.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) to
avoid breaking SSE streaming responses.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog
from starlette.types import ASGIApp, Receive, Scope, Send

if TYPE_CHECKING:
    from collections.abc import MutableMapping

from src.monitoring.logger import get_logger, request_id_ctx_var


class RequestLoggerMiddleware:
    """Pure ASGI middleware for request ID propagation and request logging."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process an ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate request ID
        headers = dict(scope.get("headers", []))
        request_id_header = headers.get(b"x-request-id")
        request_id = request_id_header.decode("utf-8") if request_id_header else str(uuid.uuid4())

        # Bind to contextvars
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request_id_ctx_var.set(request_id)

        method = scope.get("method", "")
        path = scope.get("path", "")

        logger = get_logger()
        logger.info("request_started", method=method, path=path)

        start_time = time.monotonic()
        status_code: int | None = None

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Inject X-Request-ID into response headers
                response_headers: list[Any] = list(message.get("headers", []))
                response_headers.append([b"x-request-id", request_id.encode("utf-8")])
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            logger.info(
                "request_completed",
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
            )
            structlog.contextvars.clear_contextvars()
            request_id_ctx_var.set(None)

"""ASGI middleware for PII detection and masking.

Intercepts HTTP request and response bodies to detect and mask
personally identifiable information (PII) using :class:`PIIDetector`.

Implemented as a pure ASGI middleware (not ``BaseHTTPMiddleware``) to
avoid breaking SSE streaming responses.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

from src.monitoring.logger import get_logger
from src.security.pii_detector import PIIDetector


class PIIFilterMiddleware:
    """Pure ASGI middleware for PII masking in request/response bodies.

    Args:
        app: The inner ASGI application.
        enabled: Whether PII filtering is active.
    """

    def __init__(self, app: ASGIApp, enabled: bool = True) -> None:
        self.app = app
        self._enabled = enabled
        self._detector = PIIDetector()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process an ASGI connection."""
        if scope["type"] != "http" or not self._enabled:
            await self.app(scope, receive, send)
            return

        logger = get_logger()

        async def receive_wrapper() -> MutableMapping[str, Any]:
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    text = body.decode("utf-8", errors="replace")
                    result = self._detector.detect(text)
                    if result.has_pii:
                        pii_types = [m.pii_type.value for m in result.matches]
                        logger.warning(
                            "pii_detected_in_request",
                            pii_types=pii_types,
                        )
                        message = {**message, "body": result.masked_text.encode("utf-8")}
            return message

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    text = body.decode("utf-8", errors="replace")
                    result = self._detector.detect(text)
                    if result.has_pii:
                        pii_types = [m.pii_type.value for m in result.matches]
                        logger.warning(
                            "pii_detected_in_response",
                            pii_types=pii_types,
                        )
                        message = {**message, "body": result.masked_text.encode("utf-8")}
            await send(message)

        await self.app(scope, receive_wrapper, send_wrapper)

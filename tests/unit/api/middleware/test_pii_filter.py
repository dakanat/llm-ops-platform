"""PIIフィルターミドルウェアのユニットテスト。

リクエストボディ・レスポンスボディのPIIマスキング、
無効時のパススルー、非HTTPスコープのパススルーを検証する。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

from src.api.middleware.pii_filter import PIIFilterMiddleware
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from starlette.types import Receive, Scope, Send

# --- Helpers ---


def _make_app(
    *,
    enabled: bool = True,
    handler: Any = None,
) -> Starlette:
    """テスト用Starletteアプリを構築する。"""

    async def default_handler(request: Request) -> JSONResponse:
        body = await request.body()
        text = body.decode("utf-8") if body else ""
        return JSONResponse({"echo": text})

    async def json_handler(request: Request) -> JSONResponse:
        data = await request.json()
        return JSONResponse({"received": data})

    if handler is None:
        handler = default_handler

    app = Starlette(
        routes=[
            Route("/echo", handler, methods=["POST"]),
            Route("/json", json_handler, methods=["POST"]),
        ]
    )
    app.add_middleware(PIIFilterMiddleware, enabled=enabled)  # type: ignore[arg-type,unused-ignore]
    return app


# --- PIIFilterInit ---


class TestPIIFilterInit:
    """ミドルウェア初期化のテスト。"""

    def test_creation_with_defaults(self) -> None:
        mock_app = AsyncMock()
        middleware = PIIFilterMiddleware(mock_app)
        assert middleware.app is mock_app
        assert middleware._enabled is True

    def test_creation_with_disabled(self) -> None:
        mock_app = AsyncMock()
        middleware = PIIFilterMiddleware(mock_app, enabled=False)
        assert middleware._enabled is False


# --- Non-HTTP Passthrough ---


class TestNonHttpPassthrough:
    """HTTP以外のスコープはそのまま通過する。"""

    async def test_lifespan_scope_passes_through(self) -> None:
        inner_called = False

        async def mock_app(scope: Scope, receive: Receive, send: Send) -> None:
            nonlocal inner_called
            inner_called = True

        middleware = PIIFilterMiddleware(mock_app)
        await middleware({"type": "lifespan"}, AsyncMock(), AsyncMock())
        assert inner_called


# --- Disabled Mode ---


class TestDisabledMode:
    """無効時はリクエスト・レスポンスを変更しない。"""

    def test_request_body_not_modified(self) -> None:
        app = _make_app(enabled=False)
        client = TestClient(app)
        response = client.post("/echo", content="user@example.com")
        data = response.json()
        assert data["echo"] == "user@example.com"

    def test_response_body_not_modified(self) -> None:
        async def pii_response_handler(request: Request) -> JSONResponse:
            return JSONResponse({"email": "user@example.com"})

        app = _make_app(enabled=False, handler=pii_response_handler)
        client = TestClient(app)
        response = client.post("/echo")
        data = response.json()
        assert data["email"] == "user@example.com"


# --- Request Body Masking ---


class TestRequestBodyMasking:
    """リクエストボディのPIIマスキング。"""

    def test_masks_pii_in_request(self) -> None:
        app = _make_app(enabled=True)
        client = TestClient(app)
        response = client.post("/echo", content="連絡先: user@example.com")
        data = response.json()
        assert "user@example.com" not in data["echo"]
        assert "[EMAIL]" in data["echo"]

    def test_no_change_without_pii(self) -> None:
        app = _make_app(enabled=True)
        client = TestClient(app)
        response = client.post("/echo", content="hello world")
        data = response.json()
        assert data["echo"] == "hello world"

    def test_masks_phone_in_json_request(self) -> None:
        app = _make_app(enabled=True)
        client = TestClient(app)
        payload = json.dumps({"phone": "090-1234-5678"})
        response = client.post("/echo", content=payload)
        data = response.json()
        assert "090-1234-5678" not in data["echo"]
        assert "[PHONE]" in data["echo"]


# --- Response Body Masking ---


class TestResponseBodyMasking:
    """レスポンスボディのPIIマスキング。"""

    def test_masks_pii_in_response(self) -> None:
        async def pii_handler(request: Request) -> JSONResponse:
            return JSONResponse({"result": "Contact user@example.com for info"})

        app = _make_app(enabled=True, handler=pii_handler)
        client = TestClient(app)
        response = client.post("/echo")
        body = response.text
        assert "user@example.com" not in body
        assert "[EMAIL]" in body

    def test_no_change_without_pii_in_response(self) -> None:
        async def clean_handler(request: Request) -> JSONResponse:
            return JSONResponse({"result": "no pii here"})

        app = _make_app(enabled=True, handler=clean_handler)
        client = TestClient(app)
        response = client.post("/echo")
        data = response.json()
        assert data["result"] == "no pii here"


# --- Streaming Response Masking ---


class TestStreamingResponseMasking:
    """ストリーミングレスポンスのPIIマスキング。"""

    def test_each_sse_chunk_masked_independently(self) -> None:
        async def sse_handler(request: Request) -> StreamingResponse:
            async def generate() -> AsyncIterator[str]:
                yield "data: user@example.com\n\n"
                yield "data: 090-1234-5678\n\n"
                yield "data: clean text\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        app = _make_app(enabled=True, handler=sse_handler)
        client = TestClient(app)
        response = client.post("/echo")
        body = response.text
        assert "user@example.com" not in body
        assert "090-1234-5678" not in body
        assert "[EMAIL]" in body
        assert "[PHONE]" in body
        assert "clean text" in body

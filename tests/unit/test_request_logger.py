"""Tests for ASGI request logger middleware."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock

import pytest
import structlog
from httpx import ASGITransport, AsyncClient
from src.api.middleware.request_logger import RequestLoggerMiddleware
from src.main import app
from src.monitoring.logger import request_id_ctx_var, setup_logging
from starlette.types import Receive, Scope, Send


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    """Reset structlog and dependency overrides after each test."""
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    request_id_ctx_var.set(None)
    yield
    app.dependency_overrides.clear()
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    request_id_ctx_var.set(None)


def _setup_log_capture() -> StringIO:
    """Setup logging with a StringIO handler for capturing JSON log output."""
    setup_logging(log_level="DEBUG")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    if root.handlers:
        handler.setFormatter(root.handlers[0].formatter)
    root.addHandler(handler)
    return stream


def _parse_log_lines(stream: StringIO) -> list[dict[str, Any]]:
    """Parse all JSON log lines from the captured stream."""
    output = stream.getvalue().strip()
    if not output:
        return []
    lines = output.split("\n")
    result: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if line:
            result.append(json.loads(line))
    return result


def _find_log_event(logs: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    """Find the first log entry matching the given event name."""
    for log in logs:
        if log.get("event") == event:
            return log
    return None


# =============================================================================
# Request ID generation
# =============================================================================


class TestRequestIdGeneration:
    """リクエストIDの生成・ヘッダー伝播。"""

    async def test_generates_request_id_when_no_header(self) -> None:
        """X-Request-ID ヘッダーがない場合にUUIDが生成されること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        request_id = response.headers.get("x-request-id")
        assert request_id is not None
        assert len(request_id) == 36  # UUID4 format

    async def test_uses_provided_request_id_header(self) -> None:
        """X-Request-ID ヘッダーが提供された場合にそれを使用すること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health", headers={"x-request-id": "custom-id-123"})

        assert response.headers.get("x-request-id") == "custom-id-123"

    async def test_response_contains_x_request_id_header(self) -> None:
        """レスポンスに X-Request-ID ヘッダーが含まれること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert "x-request-id" in response.headers


# =============================================================================
# Request logging
# =============================================================================


class TestRequestLogging:
    """リクエスト開始・完了ログの出力。"""

    async def test_logs_request_started_event(self) -> None:
        """request_started イベントがログ出力されること。"""
        stream = _setup_log_capture()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health")

        logs = _parse_log_lines(stream)
        started = _find_log_event(logs, "request_started")
        assert started is not None

    async def test_logs_request_completed_event(self) -> None:
        """request_completed イベントがログ出力されること。"""
        stream = _setup_log_capture()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health")

        logs = _parse_log_lines(stream)
        completed = _find_log_event(logs, "request_completed")
        assert completed is not None

    async def test_request_started_contains_method_and_path(self) -> None:
        """request_started ログに method と path が含まれること。"""
        stream = _setup_log_capture()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health")

        logs = _parse_log_lines(stream)
        started = _find_log_event(logs, "request_started")
        assert started is not None
        assert started["method"] == "GET"
        assert started["path"] == "/health"

    async def test_request_completed_contains_status_code(self) -> None:
        """request_completed ログに status_code が含まれること。"""
        stream = _setup_log_capture()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health")

        logs = _parse_log_lines(stream)
        completed = _find_log_event(logs, "request_completed")
        assert completed is not None
        assert completed["status_code"] == 200

    async def test_request_completed_contains_duration_ms(self) -> None:
        """request_completed ログに duration_ms が含まれること。"""
        stream = _setup_log_capture()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health")

        logs = _parse_log_lines(stream)
        completed = _find_log_event(logs, "request_completed")
        assert completed is not None
        assert "duration_ms" in completed

    async def test_duration_ms_is_positive_number(self) -> None:
        """duration_ms が正の数値であること。"""
        stream = _setup_log_capture()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health")

        logs = _parse_log_lines(stream)
        completed = _find_log_event(logs, "request_completed")
        assert completed is not None
        assert isinstance(completed["duration_ms"], (int, float))
        assert completed["duration_ms"] >= 0


# =============================================================================
# Context vars propagation
# =============================================================================


class TestContextVarsPropagation:
    """リクエストスコープでの contextvars 伝播。"""

    async def test_request_id_in_context_during_request(self) -> None:
        """リクエスト処理中に request_id が contextvars にバインドされること。"""
        captured_request_id: str | None = None

        from fastapi import APIRouter

        temp_router = APIRouter()

        @temp_router.get("/test-ctx")
        async def _test_ctx() -> dict[str, str | None]:
            nonlocal captured_request_id
            captured_request_id = request_id_ctx_var.get()
            return {"request_id": captured_request_id}

        app.include_router(temp_router)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/test-ctx", headers={"x-request-id": "ctx-test-id"})
            assert response.status_code == 200
            assert captured_request_id == "ctx-test-id"
        finally:
            # Remove the temporary route
            app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != "/test-ctx"]

    async def test_request_id_cleared_after_request(self) -> None:
        """リクエスト完了後に request_id がクリアされること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health", headers={"x-request-id": "temp-id"})

        assert request_id_ctx_var.get() is None


# =============================================================================
# Non-HTTP scopes
# =============================================================================


class TestNonHttpScopes:
    """HTTP 以外のスコープ (WebSocket 等) のパススルー。"""

    async def test_non_http_scope_passes_through(self) -> None:
        """HTTP 以外のスコープはミドルウェアをバイパスすること。"""
        inner_called = False

        async def mock_app(scope: Scope, receive: Receive, send: Send) -> None:
            nonlocal inner_called
            inner_called = True

        middleware = RequestLoggerMiddleware(mock_app)
        await middleware({"type": "lifespan"}, AsyncMock(), AsyncMock())
        assert inner_called


# =============================================================================
# Integration with existing endpoints
# =============================================================================


class TestMiddlewareWithExistingEndpoints:
    """既存エンドポイントとの統合。"""

    async def test_health_endpoint_returns_200_with_request_id(self) -> None:
        """GET /health が 200 を返し X-Request-ID ヘッダーが含まれること。"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert "x-request-id" in response.headers

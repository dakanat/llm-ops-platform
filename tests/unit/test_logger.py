"""Tests for structured logging setup (structlog + JSON output)."""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import Any

import pytest
import structlog
from src.monitoring.logger import get_logger, request_id_ctx_var, setup_logging
from structlog.testing import capture_logs


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog configuration and contextvars after each test."""
    structlog.contextvars.clear_contextvars()
    request_id_ctx_var.set(None)
    # Reset structlog to avoid cached loggers between tests
    structlog.reset_defaults()


# =============================================================================
# setup_logging
# =============================================================================


class TestSetupLogging:
    """setup_logging() が正しく structlog を構成すること。"""

    def test_configures_structlog_without_error(self) -> None:
        """setup_logging() がエラーなく完了すること。"""
        setup_logging(log_level="INFO")

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_accepts_valid_log_levels(self, level: str) -> None:
        """有効なログレベルを受け付けること。"""
        setup_logging(log_level=level)


# =============================================================================
# get_logger
# =============================================================================


class TestGetLogger:
    """get_logger() が BoundLogger を返すこと。"""

    def test_returns_bound_logger_instance(self) -> None:
        """get_logger() が BoundLogger インスタンスを返すこと。"""
        setup_logging(log_level="INFO")
        logger = get_logger()
        assert logger is not None

    def test_logger_with_initial_bindings(self) -> None:
        """初期バインディング付きでロガーを取得できること。"""
        setup_logging(log_level="INFO")
        logger = get_logger(component="test")
        with capture_logs() as cap:
            logger.info("hello")
        assert cap[0]["component"] == "test"


# =============================================================================
# Log output format (JSON)
# =============================================================================


def _capture_json_log(log_level: str = "DEBUG") -> dict[str, Any]:
    """Setup logging, emit one log, return parsed JSON from captured output."""
    setup_logging(log_level=log_level)

    # Attach a StringIO handler with the same formatter as root
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    # Copy the formatter from the setup_logging-configured handler
    if root.handlers:
        handler.setFormatter(root.handlers[0].formatter)
    root.addHandler(handler)

    try:
        logger = get_logger()
        logger.info("test_event", key="value")

        output = stream.getvalue().strip()
        last_line = output.split("\n")[-1]
        parsed: dict[str, Any] = json.loads(last_line)
        return parsed
    finally:
        root.removeHandler(handler)


class TestLogOutputFormat:
    """ログ出力が JSON 形式で必要なフィールドを含むこと。"""

    def test_log_output_is_valid_json(self) -> None:
        """ログ出力が有効な JSON であること。"""
        data = _capture_json_log()
        assert isinstance(data, dict)

    def test_log_contains_level_field(self) -> None:
        """ログに level フィールドが含まれること。"""
        data = _capture_json_log()
        assert "level" in data

    def test_log_contains_timestamp_field(self) -> None:
        """ログに timestamp フィールドが含まれること。"""
        data = _capture_json_log()
        assert "timestamp" in data

    def test_log_contains_event_field(self) -> None:
        """ログに event フィールドが含まれること。"""
        data = _capture_json_log()
        assert data["event"] == "test_event"


# =============================================================================
# contextvars integration
# =============================================================================


def _capture_json_log_with_contextvars() -> dict[str, Any]:
    """Emit a log with contextvars bound and return parsed JSON."""
    setup_logging(log_level="DEBUG")

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    if root.handlers:
        handler.setFormatter(root.handlers[0].formatter)
    root.addHandler(handler)

    try:
        logger = get_logger()
        logger.info("ctx_event")

        output = stream.getvalue().strip()
        last_line = output.split("\n")[-1]
        parsed: dict[str, Any] = json.loads(last_line)
        return parsed
    finally:
        root.removeHandler(handler)


class TestContextVarsIntegration:
    """structlog contextvars にバインドした request_id がログに含まれること。"""

    def test_request_id_appears_in_log_when_bound(self) -> None:
        """request_id をバインドするとログに含まれること。"""
        structlog.contextvars.bind_contextvars(request_id="abc-123")
        data = _capture_json_log_with_contextvars()
        assert data["request_id"] == "abc-123"

    def test_request_id_absent_when_not_bound(self) -> None:
        """request_id をバインドしていない場合にログに含まれないこと。"""
        data = _capture_json_log_with_contextvars()
        assert "request_id" not in data

    def test_clear_contextvars_removes_request_id(self) -> None:
        """clear_contextvars() で request_id が除去されること。"""
        structlog.contextvars.bind_contextvars(request_id="abc-123")
        structlog.contextvars.clear_contextvars()
        data = _capture_json_log_with_contextvars()
        assert "request_id" not in data


# =============================================================================
# request_id_ctx_var
# =============================================================================


class TestRequestIdContextVar:
    """request_id_ctx_var のデフォルト値と操作。"""

    def test_default_value_is_none(self) -> None:
        """デフォルト値が None であること。"""
        assert request_id_ctx_var.get() is None

    def test_set_and_get_request_id(self) -> None:
        """set() / get() で値を読み書きできること。"""
        request_id_ctx_var.set("req-456")
        assert request_id_ctx_var.get() == "req-456"

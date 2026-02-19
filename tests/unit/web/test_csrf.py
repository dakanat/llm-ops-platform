"""Tests for CSRF token generation and validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from src.web.csrf import generate_csrf_token, validate_csrf_token


class TestGenerateCsrfToken:
    """Tests for generate_csrf_token."""

    def test_returns_non_empty_string(self) -> None:
        token = generate_csrf_token("test-secret")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_different_calls_produce_different_tokens(self) -> None:
        token1 = generate_csrf_token("test-secret")
        token2 = generate_csrf_token("test-secret")
        assert token1 != token2

    def test_token_contains_separator(self) -> None:
        token = generate_csrf_token("test-secret")
        assert "." in token


class TestValidateCsrfToken:
    """Tests for validate_csrf_token."""

    def test_valid_token_passes(self) -> None:
        secret = "test-secret"
        token = generate_csrf_token(secret)
        request = MagicMock()
        request.cookies = {"csrf_token": token}
        request.headers = {"X-CSRF-Token": token}
        validate_csrf_token(request, secret)

    def test_missing_cookie_raises(self) -> None:
        request = MagicMock()
        request.cookies = {}
        request.headers = {"X-CSRF-Token": "some-token"}
        with pytest.raises(HTTPException) as exc_info:
            validate_csrf_token(request, "test-secret")
        assert exc_info.value.status_code == 403

    def test_missing_header_raises(self) -> None:
        secret = "test-secret"
        token = generate_csrf_token(secret)
        request = MagicMock()
        request.cookies = {"csrf_token": token}
        request.headers = {}
        with pytest.raises(HTTPException) as exc_info:
            validate_csrf_token(request, secret)
        assert exc_info.value.status_code == 403

    def test_mismatched_tokens_raises(self) -> None:
        secret = "test-secret"
        token1 = generate_csrf_token(secret)
        token2 = generate_csrf_token(secret)
        request = MagicMock()
        request.cookies = {"csrf_token": token1}
        request.headers = {"X-CSRF-Token": token2}
        with pytest.raises(HTTPException) as exc_info:
            validate_csrf_token(request, secret)
        assert exc_info.value.status_code == 403

    def test_tampered_token_raises(self) -> None:
        secret = "test-secret"
        token = generate_csrf_token(secret)
        request = MagicMock()
        request.cookies = {"csrf_token": token + "tampered"}
        request.headers = {"X-CSRF-Token": token + "tampered"}
        with pytest.raises(HTTPException) as exc_info:
            validate_csrf_token(request, secret)
        assert exc_info.value.status_code == 403

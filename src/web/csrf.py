"""CSRF double-submit cookie protection."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, status
from itsdangerous import BadSignature, URLSafeTimedSerializer


def generate_csrf_token(secret_key: str) -> str:
    """Generate a signed CSRF token.

    Args:
        secret_key: Secret key for signing.

    Returns:
        Signed token string.
    """
    serializer = URLSafeTimedSerializer(secret_key)
    token: str = serializer.dumps(secrets.token_hex(16))
    return token


def validate_csrf_token(request: Request, secret_key: str) -> None:
    """Validate CSRF token from cookie and header match.

    Uses double-submit cookie pattern:
    1. Cookie ``csrf_token`` contains the signed token.
    2. Header ``X-CSRF-Token`` must carry the same value.

    Args:
        request: Incoming request.
        secret_key: Secret key used for signing.

    Raises:
        HTTPException: If tokens are missing, mismatched, or invalid.
    """
    cookie_token = request.cookies.get("csrf_token")
    header_token = request.headers.get("X-CSRF-Token")

    if not cookie_token or not header_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing",
        )

    if cookie_token != header_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token mismatch",
        )

    serializer = URLSafeTimedSerializer(secret_key)
    try:
        serializer.loads(cookie_token, max_age=3600)
    except BadSignature:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token",
        ) from None

"""Web-specific FastAPI dependencies."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, Response
from jose import JWTError

from src.api.middleware.auth import TokenPayload, create_access_token, decode_access_token
from src.config import Settings
from src.web.csrf import validate_csrf_token


class WebAuthRedirect(HTTPException):
    """Raised when a web user is not authenticated.

    Carries ``is_htmx`` so the exception handler can decide
    between a normal redirect and an HX-Redirect header.
    """

    def __init__(self, *, is_htmx: bool = False) -> None:
        self.is_htmx = is_htmx
        super().__init__(status_code=401, detail="Not authenticated")


def _get_settings() -> Settings:
    """Return application settings."""
    return Settings()


async def get_current_web_user(
    request: Request,
    response: Response,
    settings: Annotated[Settings, Depends(_get_settings)],
) -> TokenPayload:
    """Extract the current user from the access_token cookie.

    Implements a sliding session: when the token's remaining TTL drops
    below half of the configured expiry, a fresh token is set as a cookie
    on the response so the session is transparently extended.

    Args:
        request: The incoming HTTP request.
        response: The outgoing HTTP response (for setting refresh cookies).
        settings: Application settings.

    Returns:
        TokenPayload for authenticated users.

    Raises:
        WebAuthRedirect: When the user is not authenticated.
    """
    token = request.cookies.get("access_token")
    is_htmx = bool(request.headers.get("HX-Request"))

    if not token:
        raise WebAuthRedirect(is_htmx=is_htmx)

    try:
        payload = decode_access_token(token, settings)
    except JWTError:
        raise WebAuthRedirect(is_htmx=is_htmx) from None

    remaining_seconds = (payload.exp - datetime.now(UTC)).total_seconds()
    threshold = settings.jwt_access_token_expire_minutes * 60 / 2

    if remaining_seconds < threshold:
        new_token = create_access_token(
            user_id=UUID(payload.sub),
            email=payload.email,
            role=payload.role,
            settings=settings,
        )
        response.set_cookie(
            key=settings.session_cookie_name,
            value=new_token,
            httponly=True,
            secure=settings.session_cookie_secure,
            samesite="lax",
            max_age=settings.jwt_access_token_expire_minutes * 60,
        )

    return payload


async def require_csrf(
    request: Request,
    settings: Annotated[Settings, Depends(_get_settings)],
) -> None:
    """Validate CSRF token on mutating requests.

    Args:
        request: The incoming HTTP request.
        settings: Application settings.
    """
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        validate_csrf_token(request, settings.csrf_secret_key)


CurrentWebUser = Annotated[TokenPayload, Depends(get_current_web_user)]

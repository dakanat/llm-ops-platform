"""Web-specific FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from jose import JWTError

from src.api.middleware.auth import TokenPayload, decode_access_token
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
    settings: Annotated[Settings, Depends(_get_settings)],
) -> TokenPayload:
    """Extract the current user from the access_token cookie.

    Args:
        request: The incoming HTTP request.
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
        return decode_access_token(token, settings)
    except JWTError:
        raise WebAuthRedirect(is_htmx=is_htmx) from None


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

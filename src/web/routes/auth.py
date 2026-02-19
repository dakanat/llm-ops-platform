"""Web authentication routes (login/logout)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.api.middleware.auth import create_access_token, verify_password
from src.config import Settings
from src.db.models import User
from src.web.csrf import generate_csrf_token
from src.web.templates import templates

router = APIRouter(prefix="/web")


async def _get_session() -> AsyncSession:
    """Get an async database session.

    Separated as a standalone function for easy patching in tests.
    """
    from sqlmodel.ext.asyncio.session import AsyncSession as _AsyncSession

    from src.db.session import engine

    return _AsyncSession(engine)


def _render_login(request: Request, settings: Settings, error: str | None = None) -> Response:
    """Render the login page with a fresh CSRF token."""
    csrf_token = generate_csrf_token(settings.csrf_secret_key)
    response = templates.TemplateResponse(
        request,
        "auth/login.html",
        {"csrf_token": csrf_token, "error": error},
    )
    response.set_cookie("csrf_token", csrf_token, httponly=False, samesite="strict")
    return response


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Response:
    """Display the login page."""
    settings = Settings()
    return _render_login(request, settings)


@router.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request) -> Response:
    """Process login form submission."""
    settings = Settings()
    form = await request.form()
    email = str(form.get("email", ""))
    password = str(form.get("password", ""))

    session = await _get_session()
    try:
        stmt = select(User).where(User.email == email)
        result = await session.exec(stmt)
        user: Any = result.first()
    finally:
        await session.close()

    if user is None or not verify_password(password, user.hashed_password):
        return _render_login(request, settings, error="Invalid email or password")

    if not user.is_active:
        return _render_login(request, settings, error="Invalid email or password")

    token = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
        settings=settings,
    )
    response = RedirectResponse(url="/web/chat", status_code=303)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.jwt_access_token_expire_minutes * 60,
    )
    return response


@router.post("/logout")
async def logout(request: Request) -> Response:
    """Clear the session cookie and redirect to login."""
    settings = Settings()
    response = RedirectResponse(url="/web/login", status_code=303)
    response.delete_cookie(key=settings.session_cookie_name)
    return response

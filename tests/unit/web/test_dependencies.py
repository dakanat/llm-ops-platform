"""Tests for web authentication dependencies."""

from __future__ import annotations

from uuid import UUID

from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from src.api.middleware.auth import TokenPayload, create_access_token
from src.web.dependencies import WebAuthRedirect, get_current_web_user
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, RedirectResponse


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with the web auth dependency and handler."""
    app = FastAPI()

    @app.exception_handler(WebAuthRedirect)
    async def _handler(
        request: StarletteRequest, exc: WebAuthRedirect
    ) -> JSONResponse | RedirectResponse:
        if exc.is_htmx:
            return JSONResponse(
                content="",
                status_code=200,
                headers={"HX-Redirect": "/web/login"},
            )
        return RedirectResponse(url="/web/login", status_code=303)

    _web_user_dep = Depends(get_current_web_user)

    @app.get("/test")
    async def test_route(
        user: TokenPayload = _web_user_dep,  # noqa: B008
    ) -> dict[str, str]:
        return {"email": user.email, "role": user.role}

    return app


class TestGetCurrentWebUser:
    """Tests for get_current_web_user."""

    async def test_valid_cookie_returns_user(self) -> None:
        token = create_access_token(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            email="test@example.com",
            role="user",
        )
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            client.cookies.set("access_token", token)
            resp = await client.get("/test")
            assert resp.status_code == 200
            data = resp.json()
            assert data["email"] == "test@example.com"
            assert data["role"] == "user"

    async def test_missing_cookie_redirects_to_login(self) -> None:
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            resp = await client.get("/test")
            assert resp.status_code == 303
            assert "/web/login" in resp.headers["location"]

    async def test_missing_cookie_htmx_returns_hx_redirect(self) -> None:
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/test", headers={"HX-Request": "true"})
            assert resp.status_code == 200
            assert resp.headers.get("HX-Redirect") == "/web/login"

    async def test_invalid_token_redirects_to_login(self) -> None:
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            client.cookies.set("access_token", "invalid-token")
            resp = await client.get("/test")
            assert resp.status_code == 303
            assert "/web/login" in resp.headers["location"]

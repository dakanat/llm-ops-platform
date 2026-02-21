"""Tests for web authentication dependencies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from jose import jwt
from src.api.middleware.auth import TokenPayload, create_access_token
from src.config import Settings
from src.web.dependencies import WebAuthRedirect, get_current_web_user
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, RedirectResponse


def _create_token_with_ttl(minutes_remaining: int) -> str:
    """Create a JWT token that expires in the given number of minutes."""
    settings = Settings()
    expire = datetime.now(UTC) + timedelta(minutes=minutes_remaining)
    claims = {
        "sub": "00000000-0000-0000-0000-000000000001",
        "email": "test@example.com",
        "role": "user",
        "exp": expire,
    }
    encoded: str = jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded


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

    async def test_token_near_expiry_gets_refreshed(self) -> None:
        """Token with <15 min remaining should get a refreshed cookie."""
        token = _create_token_with_ttl(minutes_remaining=10)
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            client.cookies.set("access_token", token)
            resp = await client.get("/test")
            assert resp.status_code == 200
            assert "set-cookie" in resp.headers
            assert "access_token=" in resp.headers["set-cookie"]

    async def test_token_far_from_expiry_not_refreshed(self) -> None:
        """Token with plenty of time remaining should NOT get a refreshed cookie."""
        token = _create_token_with_ttl(minutes_remaining=25)
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            client.cookies.set("access_token", token)
            resp = await client.get("/test")
            assert resp.status_code == 200
            assert "access_token=" not in resp.headers.get("set-cookie", "")

    async def test_expired_token_still_redirects(self) -> None:
        """Expired token should still redirect to login (not refresh)."""
        token = _create_token_with_ttl(minutes_remaining=-1)
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            follow_redirects=False,
        ) as client:
            client.cookies.set("access_token", token)
            resp = await client.get("/test")
            assert resp.status_code == 303
            assert "/web/login" in resp.headers["location"]

    async def test_token_near_expiry_refreshed_on_htmx_request(self) -> None:
        """HTMX requests should also get refreshed cookies."""
        token = _create_token_with_ttl(minutes_remaining=10)
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            client.cookies.set("access_token", token)
            resp = await client.get("/test", headers={"HX-Request": "true"})
            assert resp.status_code == 200
            assert "set-cookie" in resp.headers
            assert "access_token=" in resp.headers["set-cookie"]

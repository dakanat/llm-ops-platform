"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.request_logger import RequestLoggerMiddleware
from src.api.routes.admin import router as admin_router
from src.api.routes.agent import router as agent_router
from src.api.routes.chat import router as chat_router
from src.api.routes.eval import router as eval_router
from src.api.routes.rag import router as rag_router
from src.config import Settings
from src.monitoring.logger import setup_logging
from src.web.routes.auth import router as web_auth_router


def _register_web_exception_handlers(application: FastAPI) -> None:
    """Register exception handlers for web authentication redirects."""
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse, RedirectResponse

    from src.web.dependencies import WebAuthRedirect

    @application.exception_handler(WebAuthRedirect)
    async def web_auth_redirect_handler(
        request: StarletteRequest, exc: WebAuthRedirect
    ) -> JSONResponse | RedirectResponse:
        if exc.is_htmx:
            return JSONResponse(
                content="",
                status_code=200,
                headers={"HX-Redirect": "/web/login"},
            )
        return RedirectResponse(url="/web/login", status_code=303)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a FastAPI application.

    Args:
        settings: Application settings. If None, a new Settings instance is
            created from environment variables.

    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = Settings()
    setup_logging(
        log_level=settings.log_level,
        pii_mask_logs=settings.pii_detection_enabled and settings.pii_mask_logs,
    )

    application = FastAPI(
        title="LLM Ops Platform",
        description="Production-ready LLM application platform",
        version="0.1.0",
    )

    # Create Redis client for rate limiting (only when enabled)
    _rate_limit_redis = None
    if settings.rate_limit_enabled:
        from redis.asyncio import Redis

        _rate_limit_redis = Redis.from_url(settings.redis_url)

    # Middleware registration order: last added = outermost = runs first.
    # Execution order: RequestLogger -> RateLimit -> routes
    application.add_middleware(
        RateLimitMiddleware,  # type: ignore[arg-type,unused-ignore]  # Starlette typing issue
        redis_client=_rate_limit_redis,
        enabled=settings.rate_limit_enabled,
        requests_per_minute=settings.rate_limit_requests_per_minute,
        burst_size=settings.rate_limit_burst_size,
    )
    application.add_middleware(RequestLoggerMiddleware)  # type: ignore[arg-type,unused-ignore]  # Starlette typing issue

    # JSON API routers
    application.include_router(chat_router)
    application.include_router(rag_router)
    application.include_router(agent_router)
    application.include_router(eval_router)

    from src.api.routes.eval_datasets import router as eval_datasets_router

    application.include_router(eval_datasets_router)
    application.include_router(admin_router)

    from src.api.routes.auth import router as auth_router

    application.include_router(auth_router)

    # Web exception handlers
    _register_web_exception_handlers(application)

    # Web frontend routers
    application.include_router(web_auth_router)

    from src.web.routes.chat import router as web_chat_router

    application.include_router(web_chat_router)

    from src.web.routes.rag import router as web_rag_router

    application.include_router(web_rag_router)

    from src.web.routes.rag_documents import router as web_rag_documents_router

    application.include_router(web_rag_documents_router)

    from src.web.routes.agent import router as web_agent_router

    application.include_router(web_agent_router)

    from src.web.routes.eval import router as web_eval_router

    application.include_router(web_eval_router)

    from src.web.routes.eval_datasets import router as web_eval_datasets_router

    application.include_router(web_eval_datasets_router)

    from src.web.routes.admin import router as web_admin_router

    application.include_router(web_admin_router)

    # Static files
    static_dir = Path(__file__).parent / "web" / "static"
    application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @application.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @application.get("/")
    async def root_redirect() -> RedirectResponse:
        """Redirect root to web chat page."""
        return RedirectResponse(url="/web/chat", status_code=307)

    return application


app = create_app()

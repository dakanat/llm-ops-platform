"""FastAPI application entry point."""

from fastapi import FastAPI

from src.api.middleware.pii_filter import PIIFilterMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.request_logger import RequestLoggerMiddleware
from src.api.routes.admin import router as admin_router
from src.api.routes.agent import router as agent_router
from src.api.routes.chat import router as chat_router
from src.api.routes.eval import router as eval_router
from src.api.routes.rag import router as rag_router
from src.config import Settings
from src.monitoring.logger import setup_logging

settings = Settings()
setup_logging(log_level=settings.log_level)

app = FastAPI(
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
# Execution order: RequestLogger -> RateLimit -> PIIFilter -> routes
app.add_middleware(PIIFilterMiddleware, enabled=settings.pii_detection_enabled)  # type: ignore[arg-type,unused-ignore]
app.add_middleware(
    RateLimitMiddleware,  # type: ignore[arg-type,unused-ignore]  # Starlette typing issue
    redis_client=_rate_limit_redis,
    enabled=settings.rate_limit_enabled,
    requests_per_minute=settings.rate_limit_requests_per_minute,
    burst_size=settings.rate_limit_burst_size,
)
app.add_middleware(RequestLoggerMiddleware)  # type: ignore[arg-type,unused-ignore]  # Starlette typing issue

app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(eval_router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

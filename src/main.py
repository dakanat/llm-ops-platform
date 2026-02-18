"""FastAPI application entry point."""

from fastapi import FastAPI

from src.api.middleware.request_logger import RequestLoggerMiddleware
from src.api.routes.chat import router as chat_router
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

app.add_middleware(RequestLoggerMiddleware)  # type: ignore[arg-type,unused-ignore]  # Starlette typing issue

app.include_router(chat_router)
app.include_router(rag_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

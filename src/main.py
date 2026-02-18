"""FastAPI application entry point."""

from fastapi import FastAPI

from src.api.middleware.pii_filter import PIIFilterMiddleware
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

app.add_middleware(RequestLoggerMiddleware)  # type: ignore[arg-type,unused-ignore]  # Starlette typing issue
app.add_middleware(PIIFilterMiddleware, enabled=settings.pii_detection_enabled)  # type: ignore[arg-type,unused-ignore]

app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(eval_router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

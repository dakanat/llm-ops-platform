"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="LLM Ops Platform",
    description="Production-ready LLM application platform",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

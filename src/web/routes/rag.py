"""Web RAG interface routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse

from src.api.dependencies import get_rag_pipeline
from src.rag.pipeline import RAGPipeline, RAGPipelineError
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

router = APIRouter(prefix="/web")


@router.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request, user: CurrentWebUser) -> Response:
    """Display the RAG query page."""
    return templates.TemplateResponse(
        request, "rag/page.html", {"user": user, "active_page": "rag"}
    )


@router.post("/rag/query", response_class=HTMLResponse)
async def rag_query(
    request: Request,
    user: CurrentWebUser,
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
) -> Response:
    """Execute a RAG query and return the result as HTML."""
    form = await request.form()
    query = str(form.get("query", "")).strip()

    if not query:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": "Query cannot be empty"},
        )

    try:
        result = await pipeline.query(query=query)
    except RAGPipelineError as e:
        return templates.TemplateResponse(
            request,
            "components/error_toast.html",
            {"error_message": f"RAG error: {e}"},
        )

    return templates.TemplateResponse(
        request,
        "rag/result.html",
        {
            "query": query,
            "answer": result.answer,
            "sources": result.sources,
        },
    )

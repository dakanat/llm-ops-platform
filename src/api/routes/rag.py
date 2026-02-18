"""RAG query endpoint."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_llm_model, get_rag_pipeline
from src.llm.providers.base import TokenUsage
from src.rag.pipeline import RAGPipeline, RAGPipelineError
from src.rag.retriever import RetrievedChunk

router = APIRouter(prefix="/rag")


class RAGQueryRequest(BaseModel):
    """RAG query request body."""

    query: str = Field(min_length=1)
    top_k: int = 5
    document_id: uuid.UUID | None = None


class RAGQueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: list[RetrievedChunk]
    model: str
    usage: TokenUsage | None = None


@router.post("/query")
async def rag_query(
    request: RAGQueryRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
    model: Annotated[str, Depends(get_llm_model)],
) -> RAGQueryResponse:
    """Query the RAG pipeline and return an answer with sources."""
    try:
        result = await pipeline.query(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except RAGPipelineError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return RAGQueryResponse(
        answer=result.answer,
        sources=result.sources,
        model=model,
        usage=result.usage,
    )

"""Web RAG document management routes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import delete, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.db.models import Chunk, Document
from src.rag.pipeline import RAGPipeline, RAGPipelineError
from src.web.dependencies import CurrentWebUser
from src.web.templates import templates

if TYPE_CHECKING:
    from src.rag.embedder import Embedder
    from src.rag.gemini_embedder import GeminiEmbedder

router = APIRouter(prefix="/web")


async def _get_session() -> AsyncSession:
    """Get an async database session. Separated for testability."""
    from src.db.session import engine

    return AsyncSession(engine)


async def _build_rag_pipeline(session: AsyncSession) -> RAGPipeline:
    """Build a RAG pipeline sharing the given session.

    Mirrors ``get_rag_pipeline`` from ``src.api.dependencies`` but accepts
    an explicit session so Document and Chunk inserts share one transaction.
    """
    from src.config import Settings
    from src.db.vector_store import VectorStore
    from src.llm.router import LLMRouter
    from src.rag.chunker import RecursiveCharacterSplitter
    from src.rag.generator import Generator
    from src.rag.index_manager import IndexManager
    from src.rag.preprocessor import Preprocessor
    from src.rag.retriever import Retriever

    settings = Settings()

    # Create embedder based on provider setting
    embedder: Embedder | GeminiEmbedder
    if settings.embedding_provider == "gemini":
        from src.rag.gemini_embedder import GeminiEmbedder

        embedder = GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.embedding_gemini_model,
        )
    elif settings.embedding_provider == "local":
        from src.rag.embedder import Embedder

        embedder = Embedder(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )
    else:
        msg = f"Unknown embedding provider: '{settings.embedding_provider}'"
        raise ValueError(msg)

    vector_store = VectorStore(session=session)
    preprocessor = Preprocessor()
    chunker = RecursiveCharacterSplitter()
    index_manager = IndexManager(
        preprocessor=preprocessor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    provider = LLMRouter(settings=settings).get_provider()
    if settings.pii_detection_enabled and settings.pii_mask_llm_outbound:
        from src.llm.pii_sanitizing_provider import PIISanitizingProvider

        provider = PIISanitizingProvider(inner=provider)
    generator = Generator(llm_provider=provider, model=settings.llm_model)

    return RAGPipeline(
        index_manager=index_manager,
        retriever=retriever,
        generator=generator,
    )


@router.get("/rag/documents", response_class=HTMLResponse)
async def rag_documents_list(request: Request, user: CurrentWebUser) -> Response:
    """Display the document list page with chunk counts."""
    session = await _get_session()
    try:
        stmt = (
            select(
                Document,
                func.count(Chunk.id),  # type: ignore[arg-type]
            )
            .outerjoin(Chunk, Document.id == Chunk.document_id)  # type: ignore[arg-type]
            .where(Document.user_id == UUID(user.sub))
            .group_by(Document.id)  # type: ignore[arg-type]
            .order_by(Document.created_at.desc())  # type: ignore[attr-defined]
        )
        result = await session.exec(stmt)
        rows = list(result.all())
    finally:
        await session.close()

    documents = [{"document": doc, "chunk_count": count} for doc, count in rows]

    return templates.TemplateResponse(
        request,
        "rag/document_list.html",
        {"user": user, "active_page": "rag_documents", "documents": documents},
    )


@router.get("/rag/documents/create", response_class=HTMLResponse)
async def rag_document_create_form(request: Request, user: CurrentWebUser) -> Response:
    """Display the document creation form."""
    return templates.TemplateResponse(
        request,
        "rag/document_form.html",
        {"user": user, "active_page": "rag_documents"},
    )


@router.post("/rag/documents", response_class=HTMLResponse)
async def rag_document_create(request: Request, user: CurrentWebUser) -> Response:
    """Create a new document and index it via the RAG pipeline."""
    form = await request.form()
    title = str(form.get("title", "")).strip()
    content = str(form.get("content", "")).strip()

    if not title:
        return templates.TemplateResponse(
            request,
            "rag/document_form.html",
            {
                "user": user,
                "active_page": "rag_documents",
                "error_message": "Title is required.",
                "form_title": title,
                "form_content": content,
            },
        )

    if not content:
        return templates.TemplateResponse(
            request,
            "rag/document_form.html",
            {
                "user": user,
                "active_page": "rag_documents",
                "error_message": "Content is required.",
                "form_title": title,
                "form_content": content,
            },
        )

    session = await _get_session()
    try:
        document = Document(
            title=title,
            content=content,
            user_id=UUID(user.sub),
        )
        session.add(document)
        await session.flush()

        pipeline = await _build_rag_pipeline(session)
        await pipeline.index_document(document)
        await session.commit()
    except RAGPipelineError as e:
        await session.rollback()
        return templates.TemplateResponse(
            request,
            "rag/document_form.html",
            {
                "user": user,
                "active_page": "rag_documents",
                "error_message": f"Indexing error: {e}",
                "form_title": title,
                "form_content": content,
            },
        )
    finally:
        await session.close()

    return RedirectResponse(url="/web/rag/documents", status_code=303)


@router.get("/rag/documents/{document_id}", response_class=HTMLResponse)
async def rag_document_detail(document_id: str, request: Request, user: CurrentWebUser) -> Response:
    """Display a single document's details with its chunks."""
    session = await _get_session()
    try:
        doc_uuid = UUID(document_id)
        document = await session.get(Document, doc_uuid)
        if document is None:
            return templates.TemplateResponse(
                request,
                "components/error_toast.html",
                {"error_message": "Document not found"},
                status_code=404,
            )

        stmt = (
            select(Chunk).where(Chunk.document_id == doc_uuid).order_by(Chunk.chunk_index)  # type: ignore[arg-type]
        )
        result = await session.exec(stmt)
        chunks = list(result.all())
    finally:
        await session.close()

    return templates.TemplateResponse(
        request,
        "rag/document_detail.html",
        {
            "user": user,
            "active_page": "rag_documents",
            "document": document,
            "chunks": chunks,
        },
    )


@router.post("/rag/documents/{document_id}/delete", response_class=HTMLResponse)
async def rag_document_delete(document_id: str, request: Request, user: CurrentWebUser) -> Response:
    """Delete a document and its chunks."""
    session = await _get_session()
    try:
        doc_uuid = UUID(document_id)
        document = await session.get(Document, doc_uuid)
        if document is None:
            return templates.TemplateResponse(
                request,
                "components/error_toast.html",
                {"error_message": "Document not found"},
                status_code=404,
            )

        # Delete chunks first (no CASCADE on FK)
        await session.exec(delete(Chunk).where(Chunk.document_id == doc_uuid))  # type: ignore[arg-type]
        await session.delete(document)
        await session.commit()
    finally:
        await session.close()

    return RedirectResponse(url="/web/rag/documents", status_code=303)

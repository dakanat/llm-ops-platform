"""Tests for web RAG document management routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from httpx import AsyncClient

from tests.unit.web.conftest import AuthCookies

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_UUID = UUID("00000000-0000-0000-0000-000000000001")


def _make_document(
    *,
    doc_id: UUID | None = None,
    title: str = "Test Doc",
    content: str = "Hello world",
) -> MagicMock:
    """Return a mock Document-like object."""
    from datetime import UTC, datetime

    doc = MagicMock()
    doc.id = doc_id or UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    doc.title = title
    doc.content = content
    doc.user_id = _USER_UUID
    doc.created_at = datetime(2025, 1, 1, tzinfo=UTC)
    doc.updated_at = datetime(2025, 1, 1, tzinfo=UTC)
    return doc


def _make_chunk(*, chunk_index: int = 0, has_embedding: bool = True) -> MagicMock:
    """Return a mock Chunk-like object."""
    chunk = MagicMock()
    chunk.id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    chunk.content = "chunk content"
    chunk.chunk_index = chunk_index
    chunk.embedding = [0.1] * 10 if has_embedding else None
    return chunk


# ---------------------------------------------------------------------------
# Tests: GET /web/rag/documents
# ---------------------------------------------------------------------------


class TestRagDocumentsList:
    """Tests for the document list page."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        """Unauthenticated users are redirected to login."""
        resp = await client.get("/web/rag/documents")
        assert resp.status_code == 303

    @patch("src.web.routes.rag_documents._get_session")
    async def test_authenticated_returns_list(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Authenticated users see the document list page."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.exec.return_value = mock_result
        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.get("/web/rag/documents")

        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Document" in resp.text or "document" in resp.text

    @patch("src.web.routes.rag_documents._get_session")
    async def test_list_shows_document_with_chunk_count(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Document list shows documents with their chunk counts."""
        doc = _make_document(title="My Knowledge Base")
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [(doc, 5)]
        mock_session.exec.return_value = mock_result
        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.get("/web/rag/documents")

        assert resp.status_code == 200
        assert "My Knowledge Base" in resp.text
        assert "5" in resp.text


# ---------------------------------------------------------------------------
# Tests: GET /web/rag/documents/create
# ---------------------------------------------------------------------------


class TestRagDocumentCreateForm:
    """Tests for the document creation form."""

    async def test_unauthenticated_redirects(self, client: AsyncClient) -> None:
        """Unauthenticated users are redirected to login."""
        resp = await client.get("/web/rag/documents/create")
        assert resp.status_code == 303

    async def test_form_renders(
        self,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Authenticated users see the creation form."""
        with AuthCookies(client, admin_token):
            resp = await client.get("/web/rag/documents/create")

        assert resp.status_code == 200
        assert "form" in resp.text.lower()


# ---------------------------------------------------------------------------
# Tests: POST /web/rag/documents
# ---------------------------------------------------------------------------


class TestRagDocumentCreate:
    """Tests for document creation (POST)."""

    @patch("src.web.routes.rag_documents._build_rag_pipeline")
    @patch("src.web.routes.rag_documents._get_session")
    async def test_saves_and_redirects_on_success(
        self,
        mock_get_session: AsyncMock,
        mock_build_pipeline: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Successful creation saves document, indexes, and redirects."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add() is synchronous
        mock_get_session.return_value = mock_session

        mock_pipeline = AsyncMock()
        mock_pipeline.index_document.return_value = [_make_chunk()]
        mock_build_pipeline.return_value = mock_pipeline

        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/rag/documents",
                data={"title": "New Doc", "content": "Some content here"},
            )

        assert resp.status_code == 303
        assert "/web/rag/documents" in resp.headers["location"]
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()
        mock_pipeline.index_document.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @patch("src.web.routes.rag_documents._get_session")
    async def test_error_on_empty_title(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Empty title returns an error response."""
        mock_session = AsyncMock()
        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/rag/documents",
                data={"title": "", "content": "Some content"},
            )

        assert resp.status_code == 200
        assert "title" in resp.text.lower() or "Title" in resp.text

    @patch("src.web.routes.rag_documents._get_session")
    async def test_error_on_empty_content(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Empty content returns an error response."""
        mock_session = AsyncMock()
        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/rag/documents",
                data={"title": "My Doc", "content": ""},
            )

        assert resp.status_code == 200
        assert "content" in resp.text.lower() or "Content" in resp.text

    @patch("src.web.routes.rag_documents._build_rag_pipeline")
    @patch("src.web.routes.rag_documents._get_session")
    async def test_error_on_indexing_failure(
        self,
        mock_get_session: AsyncMock,
        mock_build_pipeline: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Indexing failure rolls back and shows error."""
        from src.rag.pipeline import RAGPipelineError

        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add() is synchronous
        mock_get_session.return_value = mock_session

        mock_pipeline = AsyncMock()
        mock_pipeline.index_document.side_effect = RAGPipelineError("embed failed")
        mock_build_pipeline.return_value = mock_pipeline

        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/rag/documents",
                data={"title": "Fail Doc", "content": "content"},
            )

        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "Error" in resp.text
        mock_session.rollback.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: GET /web/rag/documents/{document_id}
# ---------------------------------------------------------------------------


class TestRagDocumentDetail:
    """Tests for the document detail page."""

    @patch("src.web.routes.rag_documents._get_session")
    async def test_shows_document_and_chunks(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Detail page renders document info and chunks."""
        doc = _make_document(title="Detail Doc")
        chunks = [_make_chunk(chunk_index=0), _make_chunk(chunk_index=1)]

        mock_session = AsyncMock()
        mock_session.get.return_value = doc

        mock_result = MagicMock()
        mock_result.all.return_value = chunks
        mock_session.exec.return_value = mock_result

        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.get(
                f"/web/rag/documents/{doc.id}",
            )

        assert resp.status_code == 200
        assert "Detail Doc" in resp.text

    @patch("src.web.routes.rag_documents._get_session")
    async def test_404_on_missing_document(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Missing document returns 404."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None
        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.get(
                "/web/rag/documents/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            )

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /web/rag/documents/{document_id}/delete
# ---------------------------------------------------------------------------


class TestRagDocumentDelete:
    """Tests for document deletion."""

    @patch("src.web.routes.rag_documents._get_session")
    async def test_deletes_and_redirects(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Successful deletion removes doc and redirects."""
        doc = _make_document()
        mock_session = AsyncMock()
        mock_session.get.return_value = doc
        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.post(
                f"/web/rag/documents/{doc.id}/delete",
            )

        assert resp.status_code == 303
        assert "/web/rag/documents" in resp.headers["location"]
        mock_session.commit.assert_awaited_once()

    @patch("src.web.routes.rag_documents._get_session")
    async def test_404_on_missing_document(
        self,
        mock_get_session: AsyncMock,
        client: AsyncClient,
        admin_token: str,
    ) -> None:
        """Deleting a missing document returns 404."""
        mock_session = AsyncMock()
        mock_session.get.return_value = None
        mock_get_session.return_value = mock_session

        with AuthCookies(client, admin_token):
            resp = await client.post(
                "/web/rag/documents/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/delete",
            )

        assert resp.status_code == 404

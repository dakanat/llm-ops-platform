"""Integration tests for the RAG pipeline.

Tests document indexing, vector similarity search, and end-to-end
RAG query flow using a real PostgreSQL+pgvector database.
External services (LLM API, vLLM embedding) are replaced with
FakeEmbedder and mock LLM providers.
"""

from __future__ import annotations

import pytest
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession
from src.db.models import Chunk, Document, User
from src.db.vector_store import VectorStore
from src.rag.chunker import RecursiveCharacterSplitter
from src.rag.generator import Generator
from src.rag.index_manager import IndexManager
from src.rag.pipeline import RAGPipeline
from src.rag.preprocessor import Preprocessor
from src.rag.retriever import Retriever

from .conftest import (
    FakeEmbedder,
    make_deterministic_embedding,
    make_mock_llm_provider,
    make_similar_embedding,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")


# ===========================================================================
# TestDocumentIndexing
# ===========================================================================
class TestDocumentIndexing:
    """Tests for indexing documents through real DB with FakeEmbedder."""

    async def _build_index_manager(
        self,
        db_session: SQLModelAsyncSession,
        embedder: FakeEmbedder | None = None,
    ) -> IndexManager:
        """Build an IndexManager wired to real DB components."""
        return IndexManager(
            preprocessor=Preprocessor(),
            chunker=RecursiveCharacterSplitter(chunk_size=128, chunk_overlap=16),
            embedder=embedder or FakeEmbedder(),
            vector_store=VectorStore(session=db_session),
        )

    async def test_index_document_saves_chunks_to_database(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Chunks are persisted in the database after indexing."""
        doc = Document(
            title="Indexing Test",
            content="Alpha bravo charlie. " * 20,
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()

        manager = await self._build_index_manager(db_session)
        chunks = await manager.index_document(doc)

        assert len(chunks) > 0

        result = await db_session.exec(select(Chunk).where(Chunk.document_id == doc.id))
        db_chunks = list(result.all())
        assert len(db_chunks) == len(chunks)

    async def test_index_document_stores_embeddings_in_pgvector(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Each chunk has a non-None 1024-dim embedding stored in pgvector."""
        doc = Document(
            title="Embedding Test",
            content="Delta echo foxtrot. " * 20,
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()

        manager = await self._build_index_manager(db_session)
        chunks = await manager.index_document(doc)

        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1024

    async def test_index_document_preserves_chunk_order(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """chunk_index values are sequential starting from 0."""
        doc = Document(
            title="Order Test",
            content="Golf hotel india. " * 20,
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()

        manager = await self._build_index_manager(db_session)
        chunks = await manager.index_document(doc)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    async def test_index_document_applies_preprocessing(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Full-width characters are normalized to half-width in stored chunks."""
        # Use full-width "ＡＢＣＤ" which should be normalized to "ABCD"
        doc = Document(
            title="Preprocessing Test",
            content="\uff21\uff22\uff23\uff24 normalized text content",
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()

        manager = await self._build_index_manager(db_session)
        chunks = await manager.index_document(doc)

        # After NFKC normalization, full-width letters become ASCII
        assert all("\uff21" not in c.content for c in chunks)
        assert any("ABCD" in c.content for c in chunks)


# ===========================================================================
# TestVectorSimilaritySearch
# ===========================================================================
class TestVectorSimilaritySearch:
    """Tests for pgvector cosine similarity search with known embeddings."""

    async def test_search_returns_most_similar_chunks_first(
        self,
        db_session: SQLModelAsyncSession,
        test_document: Document,
    ) -> None:
        """The chunk with the closest embedding to the query is returned first."""
        base_emb = make_deterministic_embedding(100)
        similar_emb = make_similar_embedding(base_emb, noise=0.01)
        distant_emb = make_deterministic_embedding(999)

        chunk_close = Chunk(
            document_id=test_document.id,
            content="close chunk",
            chunk_index=0,
            embedding=similar_emb,
        )
        chunk_far = Chunk(
            document_id=test_document.id,
            content="distant chunk",
            chunk_index=1,
            embedding=distant_emb,
        )
        db_session.add_all([chunk_close, chunk_far])
        await db_session.flush()

        store = VectorStore(session=db_session)
        results = await store.search(query_embedding=base_emb, top_k=2)

        assert len(results) == 2
        assert results[0].content == "close chunk"

    async def test_search_respects_top_k_limit(
        self,
        db_session: SQLModelAsyncSession,
        test_document: Document,
    ) -> None:
        """Returns exactly top_k results when more chunks exist."""
        chunks = [
            Chunk(
                document_id=test_document.id,
                content=f"chunk {i}",
                chunk_index=i,
                embedding=make_deterministic_embedding(i),
            )
            for i in range(5)
        ]
        db_session.add_all(chunks)
        await db_session.flush()

        store = VectorStore(session=db_session)
        results = await store.search(query_embedding=make_deterministic_embedding(0), top_k=3)

        assert len(results) == 3

    async def test_search_filters_by_document_id(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Only returns chunks from the specified document."""
        doc_a = Document(title="Doc A", content="Content A", user_id=test_user.id)
        doc_b = Document(title="Doc B", content="Content B", user_id=test_user.id)
        db_session.add_all([doc_a, doc_b])
        await db_session.flush()

        chunk_a = Chunk(
            document_id=doc_a.id,
            content="chunk from A",
            chunk_index=0,
            embedding=make_deterministic_embedding(10),
        )
        chunk_b = Chunk(
            document_id=doc_b.id,
            content="chunk from B",
            chunk_index=0,
            embedding=make_deterministic_embedding(11),
        )
        db_session.add_all([chunk_a, chunk_b])
        await db_session.flush()

        store = VectorStore(session=db_session)
        results = await store.search(
            query_embedding=make_deterministic_embedding(10),
            top_k=5,
            document_id=doc_a.id,
        )

        assert all(r.document_id == doc_a.id for r in results)
        assert len(results) == 1

    async def test_search_returns_empty_when_no_chunks_exist(
        self,
        db_session: SQLModelAsyncSession,
    ) -> None:
        """Returns an empty list when there are no chunks in the database."""
        store = VectorStore(session=db_session)
        results = await store.search(query_embedding=make_deterministic_embedding(0), top_k=5)

        assert results == []


# ===========================================================================
# TestRAGPipelineEndToEnd
# ===========================================================================
class TestRAGPipelineEndToEnd:
    """End-to-end RAG pipeline tests: index → query with real DB."""

    async def _build_pipeline(
        self,
        db_session: SQLModelAsyncSession,
        embedder: FakeEmbedder,
        llm_content: str = "Generated answer",
    ) -> RAGPipeline:
        """Build a RAGPipeline with FakeEmbedder, real VectorStore, mock LLM."""
        vector_store = VectorStore(session=db_session)
        preprocessor = Preprocessor()
        chunker = RecursiveCharacterSplitter(chunk_size=128, chunk_overlap=16)
        index_manager = IndexManager(
            preprocessor=preprocessor,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )
        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        provider = make_mock_llm_provider(content=llm_content)
        generator = Generator(llm_provider=provider, model="test-model")

        return RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=generator,
        )

    async def test_index_then_query_returns_answer_with_sources(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """After indexing, a query returns a GenerationResult with answer and sources."""
        embedder = FakeEmbedder()
        pipeline = await self._build_pipeline(db_session, embedder, llm_content="The answer is 42")

        doc = Document(
            title="E2E Test",
            content="The meaning of life is forty-two. " * 10,
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()

        await pipeline.index_document(doc)
        result = await pipeline.query("What is the meaning of life?")

        assert result.answer == "The answer is 42"
        assert len(result.sources) > 0

    async def test_query_sources_reference_indexed_document(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Sources in the query result reference the indexed document's ID."""
        embedder = FakeEmbedder()
        pipeline = await self._build_pipeline(db_session, embedder)

        doc = Document(
            title="Source Reference Test",
            content="Information about quantum computing. " * 10,
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()

        await pipeline.index_document(doc)
        result = await pipeline.query("Tell me about quantum computing")

        assert all(src.document_id == doc.id for src in result.sources)

    async def test_query_with_document_id_filter(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Filtering by document_id returns only sources from that document."""
        embedder = FakeEmbedder()
        pipeline = await self._build_pipeline(db_session, embedder)

        doc_target = Document(
            title="Target Doc",
            content="Target document content about Python programming. " * 10,
            user_id=test_user.id,
        )
        doc_other = Document(
            title="Other Doc",
            content="Other document content about Java programming. " * 10,
            user_id=test_user.id,
        )
        db_session.add_all([doc_target, doc_other])
        await db_session.flush()

        await pipeline.index_document(doc_target)
        await pipeline.index_document(doc_other)

        result = await pipeline.query("Tell me about programming", document_id=doc_target.id)

        assert len(result.sources) > 0
        assert all(src.document_id == doc_target.id for src in result.sources)

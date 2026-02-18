"""Tests for POST /rag/query endpoint."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import AsyncClient
from src.llm.providers.base import TokenUsage
from src.rag.generator import GenerationResult
from src.rag.pipeline import RAGPipelineError
from src.rag.retriever import RetrievedChunk


def _make_generation_result(
    answer: str = "RAG is Retrieval-Augmented Generation.",
    model: str = "test-model",
    usage: TokenUsage | None = None,
    sources: list[RetrievedChunk] | None = None,
) -> GenerationResult:
    """Create a test GenerationResult."""
    return GenerationResult(
        answer=answer,
        model=model,
        usage=usage or TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        sources=sources
        or [
            RetrievedChunk(
                content="RAG context chunk",
                chunk_index=0,
                document_id=uuid.uuid4(),
            )
        ],
    )


def _mock_pipeline(result: GenerationResult | None = None) -> AsyncMock:
    """Create a mock RAG pipeline."""
    pipeline = AsyncMock()
    pipeline.query.return_value = result or _make_generation_result()
    return pipeline


def _override_dependencies(
    app: FastAPI,
    pipeline: AsyncMock | None = None,
    model: str = "test-model",
) -> None:
    """Set FastAPI dependency overrides for RAG route tests."""
    from src.api.dependencies import get_llm_model, get_rag_pipeline

    mock_pipeline = pipeline or _mock_pipeline()

    async def _get_pipeline() -> AsyncMock:
        return mock_pipeline

    app.dependency_overrides[get_rag_pipeline] = _get_pipeline
    app.dependency_overrides[get_llm_model] = lambda: model


class TestRAGQueryRoute:
    """POST /rag/query のテスト。"""

    async def test_returns_200_with_valid_query(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """有効なクエリで 200 が返ること。"""
        _override_dependencies(test_app)
        response = await client.post("/rag/query", json={"query": "What is RAG?"})

        assert response.status_code == 200

    async def test_returns_answer_from_pipeline(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """パイプラインの回答がレスポンスに含まれること。"""
        pipeline = _mock_pipeline(result=_make_generation_result(answer="RAG is a technique."))
        _override_dependencies(test_app, pipeline=pipeline)
        response = await client.post("/rag/query", json={"query": "What is RAG?"})

        assert response.json()["answer"] == "RAG is a technique."

    async def test_returns_sources(self, client: AsyncClient, test_app: FastAPI) -> None:
        """ソースリストがレスポンスに含まれること。"""
        doc_id = uuid.uuid4()
        sources = [
            RetrievedChunk(content="chunk text", chunk_index=0, document_id=doc_id),
        ]
        pipeline = _mock_pipeline(result=_make_generation_result(sources=sources))
        _override_dependencies(test_app, pipeline=pipeline)
        response = await client.post("/rag/query", json={"query": "What is RAG?"})

        resp_sources = response.json()["sources"]
        assert len(resp_sources) == 1
        assert resp_sources[0]["content"] == "chunk text"
        assert resp_sources[0]["chunk_index"] == 0
        assert resp_sources[0]["document_id"] == str(doc_id)

    async def test_returns_model_name(self, client: AsyncClient, test_app: FastAPI) -> None:
        """レスポンスに model フィールドが含まれること。"""
        _override_dependencies(test_app, model="openai/gpt-oss-120b:free")
        response = await client.post("/rag/query", json={"query": "What is RAG?"})

        assert response.json()["model"] == "openai/gpt-oss-120b:free"

    async def test_returns_usage(self, client: AsyncClient, test_app: FastAPI) -> None:
        """usage フィールドがレスポンスに含まれること。"""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        pipeline = _mock_pipeline(result=_make_generation_result(usage=usage))
        _override_dependencies(test_app, pipeline=pipeline)
        response = await client.post("/rag/query", json={"query": "What is RAG?"})

        resp_usage = response.json()["usage"]
        assert resp_usage["prompt_tokens"] == 100
        assert resp_usage["completion_tokens"] == 50
        assert resp_usage["total_tokens"] == 150

    async def test_passes_top_k_to_pipeline(self, client: AsyncClient, test_app: FastAPI) -> None:
        """top_k パラメータがパイプラインに渡されること。"""
        pipeline = _mock_pipeline()
        _override_dependencies(test_app, pipeline=pipeline)
        await client.post("/rag/query", json={"query": "What is RAG?", "top_k": 3})

        call_kwargs = pipeline.query.call_args.kwargs
        assert call_kwargs["top_k"] == 3

    async def test_passes_document_id_to_pipeline(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """document_id がパイプラインに渡されること。"""
        doc_id = uuid.uuid4()
        pipeline = _mock_pipeline()
        _override_dependencies(test_app, pipeline=pipeline)
        await client.post(
            "/rag/query",
            json={"query": "What is RAG?", "document_id": str(doc_id)},
        )

        call_kwargs = pipeline.query.call_args.kwargs
        assert call_kwargs["document_id"] == doc_id

    async def test_defaults_top_k_to_5(self, client: AsyncClient, test_app: FastAPI) -> None:
        """top_k 省略時にデフォルト 5 がパイプラインに渡されること。"""
        pipeline = _mock_pipeline()
        _override_dependencies(test_app, pipeline=pipeline)
        await client.post("/rag/query", json={"query": "What is RAG?"})

        call_kwargs = pipeline.query.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    async def test_returns_422_for_missing_query(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """query フィールドがない場合に 422 が返ること。"""
        _override_dependencies(test_app)
        response = await client.post("/rag/query", json={})

        assert response.status_code == 422

    async def test_returns_422_for_empty_query(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """空文字の query で 422 が返ること。"""
        _override_dependencies(test_app)
        response = await client.post("/rag/query", json={"query": ""})

        assert response.status_code == 422

    async def test_returns_502_when_pipeline_fails(
        self, client: AsyncClient, test_app: FastAPI
    ) -> None:
        """RAGPipelineError 発生時に 502 が返ること。"""
        pipeline = _mock_pipeline()
        pipeline.query.side_effect = RAGPipelineError("Pipeline failed")
        _override_dependencies(test_app, pipeline=pipeline)
        response = await client.post("/rag/query", json={"query": "What is RAG?"})

        assert response.status_code == 502

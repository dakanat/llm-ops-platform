"""SearchTool のユニットテスト。

RAG 検索ツールが RAGPipeline を通じて結果を返すことを検証する。
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from src.agent.tools.base import Tool, ToolResult
from src.agent.tools.search import SearchTool
from src.rag.generator import GenerationResult
from src.rag.pipeline import RAGPipeline, RAGPipelineError
from src.rag.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_retrieved_chunk(
    content: str = "テストチャンク",
    chunk_index: int = 0,
    document_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        chunk_index=chunk_index,
        document_id=document_id or uuid.uuid4(),
    )


def _make_generation_result(
    answer: str = "テスト回答です。",
    sources: list[RetrievedChunk] | None = None,
    model: str = "test-model",
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        sources=sources or [],
        model=model,
    )


def _make_pipeline_mock(
    generation_result: GenerationResult | None = None,
    error: RAGPipelineError | None = None,
) -> AsyncMock:
    """RAGPipeline のモックを生成する。"""
    mock = AsyncMock(spec=RAGPipeline)
    if error is not None:
        mock.query.side_effect = error
    else:
        mock.query.return_value = generation_result or _make_generation_result()
    return mock


# ---------------------------------------------------------------------------
# TestSearchToolAttributes
# ---------------------------------------------------------------------------
class TestSearchToolAttributes:
    """SearchTool の属性と Protocol 準拠を検証する。"""

    def test_name(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        assert tool.name == "search"

    def test_description_is_non_empty(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        assert len(tool.description) > 0

    def test_conforms_to_tool_protocol(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        assert isinstance(tool, Tool)


# ---------------------------------------------------------------------------
# TestSearchToolExecute
# ---------------------------------------------------------------------------
class TestSearchToolExecute:
    """SearchTool の正常系実行を検証する。"""

    @pytest.mark.asyncio
    async def test_returns_tool_result(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("テストクエリ")
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_success_has_no_error(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("テストクエリ")
        assert result.error is None
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_calls_pipeline_query(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        await tool.execute("フランスの首都は？")
        pipeline.query.assert_called_once()
        args, kwargs = pipeline.query.call_args
        assert args[0] == "フランスの首都は？"

    @pytest.mark.asyncio
    async def test_output_contains_answer(self) -> None:
        gen_result = _make_generation_result(answer="パリです。")
        pipeline = _make_pipeline_mock(generation_result=gen_result)
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("フランスの首都は？")
        assert "パリです。" in result.output

    @pytest.mark.asyncio
    async def test_output_contains_source_content(self) -> None:
        doc_id = uuid.uuid4()
        sources = [
            _make_retrieved_chunk(
                content="フランスの首都はパリ", chunk_index=0, document_id=doc_id
            ),
            _make_retrieved_chunk(content="パリはセーヌ川沿い", chunk_index=1, document_id=doc_id),
        ]
        gen_result = _make_generation_result(answer="パリです。", sources=sources)
        pipeline = _make_pipeline_mock(generation_result=gen_result)
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("フランスの首都は？")
        assert "フランスの首都はパリ" in result.output
        assert "パリはセーヌ川沿い" in result.output

    @pytest.mark.asyncio
    async def test_empty_sources_returns_answer_only(self) -> None:
        gen_result = _make_generation_result(answer="わかりません。", sources=[])
        pipeline = _make_pipeline_mock(generation_result=gen_result)
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("不明な質問")
        assert "わかりません。" in result.output
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_passes_top_k_to_pipeline(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline, top_k=3)
        await tool.execute("テストクエリ")
        _, kwargs = pipeline.query.call_args
        assert kwargs.get("top_k") == 3

    @pytest.mark.asyncio
    async def test_default_top_k_is_five(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        await tool.execute("テストクエリ")
        _, kwargs = pipeline.query.call_args
        assert kwargs.get("top_k") == 5


# ---------------------------------------------------------------------------
# TestSearchToolErrorHandling
# ---------------------------------------------------------------------------
class TestSearchToolErrorHandling:
    """SearchTool のエラー処理を検証する。例外を raise せず ToolResult で返す。"""

    @pytest.mark.asyncio
    async def test_pipeline_error_returns_error_result(self) -> None:
        pipeline = _make_pipeline_mock(error=RAGPipelineError("検索に失敗しました"))
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("テストクエリ")
        assert result.is_error is True
        assert "検索に失敗しました" in result.error  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_pipeline_error_has_empty_output(self) -> None:
        pipeline = _make_pipeline_mock(error=RAGPipelineError("エラー"))
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("テストクエリ")
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_empty_input_returns_error(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("")
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_whitespace_only_input_returns_error(self) -> None:
        pipeline = _make_pipeline_mock()
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("   ")
        assert result.is_error is True


# ---------------------------------------------------------------------------
# TestSearchToolMetadata
# ---------------------------------------------------------------------------
class TestSearchToolMetadata:
    """SearchTool の metadata.sources 構造を検証する。"""

    @pytest.mark.asyncio
    async def test_metadata_contains_sources_key(self) -> None:
        doc_id = uuid.uuid4()
        sources = [_make_retrieved_chunk(content="chunk1", chunk_index=0, document_id=doc_id)]
        gen_result = _make_generation_result(answer="answer", sources=sources)
        pipeline = _make_pipeline_mock(generation_result=gen_result)
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("query")
        assert result.metadata is not None
        assert "sources" in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_sources_structure(self) -> None:
        doc_id = uuid.uuid4()
        sources = [
            _make_retrieved_chunk(content="chunk content", chunk_index=2, document_id=doc_id),
        ]
        gen_result = _make_generation_result(answer="answer", sources=sources)
        pipeline = _make_pipeline_mock(generation_result=gen_result)
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("query")
        assert result.metadata is not None
        src = result.metadata["sources"][0]
        assert src["document_id"] == str(doc_id)
        assert src["chunk_index"] == 2
        assert src["content"] == "chunk content"

    @pytest.mark.asyncio
    async def test_metadata_sources_count_matches(self) -> None:
        doc_id = uuid.uuid4()
        sources = [
            _make_retrieved_chunk(content="a", chunk_index=0, document_id=doc_id),
            _make_retrieved_chunk(content="b", chunk_index=1, document_id=doc_id),
        ]
        gen_result = _make_generation_result(answer="answer", sources=sources)
        pipeline = _make_pipeline_mock(generation_result=gen_result)
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("query")
        assert result.metadata is not None
        assert len(result.metadata["sources"]) == 2

    @pytest.mark.asyncio
    async def test_empty_sources_yields_empty_list(self) -> None:
        gen_result = _make_generation_result(answer="no sources", sources=[])
        pipeline = _make_pipeline_mock(generation_result=gen_result)
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("query")
        assert result.metadata is not None
        assert result.metadata["sources"] == []

    @pytest.mark.asyncio
    async def test_error_result_has_no_metadata(self) -> None:
        pipeline = _make_pipeline_mock(error=RAGPipelineError("fail"))
        tool = SearchTool(pipeline=pipeline)
        result = await tool.execute("query")
        assert result.metadata is None

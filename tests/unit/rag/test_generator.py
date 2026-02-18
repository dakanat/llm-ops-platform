"""Tests for Generator (_build_context, _build_messages, generate) and GenerationResult."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.llm.providers.base import Role, TokenUsage
from src.rag.generator import GenerationError, GenerationResult, Generator

from .conftest import make_llm_response, make_retrieved_chunk


class TestGeneratorBuildContext:
    """Generator._build_context() のテスト。"""

    @pytest.fixture
    def generator(self) -> Generator:
        provider = AsyncMock()
        return Generator(llm_provider=provider, model="test-model")

    def test_builds_numbered_context(self, generator: Generator) -> None:
        """番号付きコンテキストが生成されること。"""
        chunks = [
            make_retrieved_chunk(content="チャンク1"),
            make_retrieved_chunk(content="チャンク2", chunk_index=1),
        ]

        result = generator._build_context(chunks)

        assert "[1] チャンク1" in result
        assert "[2] チャンク2" in result

    def test_separates_chunks_with_double_newline(self, generator: Generator) -> None:
        """チャンク間が二重改行で区切られること。"""
        chunks = [
            make_retrieved_chunk(content="A"),
            make_retrieved_chunk(content="B", chunk_index=1),
        ]

        result = generator._build_context(chunks)

        assert result == "[1] A\n\n[2] B"

    def test_empty_chunks_returns_empty_string(self, generator: Generator) -> None:
        """空リストで空文字列が返ること。"""
        result = generator._build_context([])

        assert result == ""


class TestGeneratorBuildMessages:
    """Generator._build_messages() のテスト。"""

    @pytest.fixture
    def generator(self) -> Generator:
        provider = AsyncMock()
        return Generator(llm_provider=provider, model="test-model")

    def test_builds_system_and_user_messages(self, generator: Generator) -> None:
        """system メッセージと user メッセージが構築されること。"""
        messages = generator._build_messages("質問テキスト", "コンテキストテキスト")

        assert len(messages) == 2
        assert messages[0].role == Role.system
        assert messages[1].role == Role.user
        assert "質問テキスト" in messages[1].content
        assert "コンテキストテキスト" in messages[1].content

    def test_uses_custom_system_prompt(self) -> None:
        """カスタムシステムプロンプトが使用されること。"""
        provider = AsyncMock()
        custom_prompt = "カスタムプロンプト"
        generator = Generator(
            llm_provider=provider, model="test-model", system_prompt=custom_prompt
        )

        messages = generator._build_messages("質問", "コンテキスト")

        assert messages[0].content == custom_prompt


class TestGeneratorGenerate:
    """Generator.generate() のテスト。"""

    @pytest.fixture
    def provider(self) -> AsyncMock:
        mock = AsyncMock()
        mock.complete.return_value = make_llm_response()
        return mock

    @pytest.fixture
    def generator(self, provider: AsyncMock) -> Generator:
        return Generator(llm_provider=provider, model="test-model")

    async def test_returns_generation_result(self, generator: Generator) -> None:
        """GenerationResult が返ること。"""
        chunks = [make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert isinstance(result, GenerationResult)

    async def test_result_contains_answer(self, generator: Generator, provider: AsyncMock) -> None:
        """回答テキストが含まれること。"""
        provider.complete.return_value = make_llm_response(content="これは回答です")
        chunks = [make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert result.answer == "これは回答です"

    async def test_result_contains_sources(self, generator: Generator) -> None:
        """ソースチャンクが含まれること。"""
        chunks = [
            make_retrieved_chunk(content="ソース1"),
            make_retrieved_chunk(content="ソース2", chunk_index=1),
        ]

        result = await generator.generate("質問", chunks)

        assert len(result.sources) == 2
        assert result.sources[0].content == "ソース1"

    async def test_result_contains_model(self, generator: Generator) -> None:
        """モデル名が含まれること。"""
        chunks = [make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert result.model == "test-model"

    async def test_result_contains_usage(self, generator: Generator, provider: AsyncMock) -> None:
        """トークン使用量が含まれること。"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        provider.complete.return_value = make_llm_response(usage=usage)
        chunks = [make_retrieved_chunk()]

        result = await generator.generate("質問", chunks)

        assert result.usage is not None
        assert result.usage.total_tokens == 30

    async def test_wraps_llm_error(self, generator: Generator, provider: AsyncMock) -> None:
        """LLM エラーが GenerationError でラップされること。"""
        llm_error = RuntimeError("LLM failed")
        provider.complete.side_effect = llm_error
        chunks = [make_retrieved_chunk()]

        with pytest.raises(GenerationError) as exc_info:
            await generator.generate("質問", chunks)

        assert exc_info.value.__cause__ is llm_error


class TestGenerationResult:
    """GenerationResult モデルのテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """必須フィールドで生成できること。"""
        chunk = make_retrieved_chunk()
        result = GenerationResult(
            answer="回答テキスト",
            sources=[chunk],
            model="test-model",
        )

        assert result.answer == "回答テキスト"
        assert len(result.sources) == 1
        assert result.model == "test-model"
        assert result.usage is None

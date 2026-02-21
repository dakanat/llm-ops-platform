"""合成データ生成のユニットテスト。

ドキュメントからQAペアを自動生成する SyntheticDataGenerator の
正常系・異常系を検証する。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.eval import EvalError, SyntheticDataError
from src.eval.datasets import EvalDataset
from src.eval.synthetic_data import QAPair, SyntheticDataGenerator
from src.llm.providers.base import LLMResponse, Role

# --- Helpers ---


def _make_mock_provider(content: str = "[]") -> AsyncMock:
    """テスト用の LLMProvider モックを作成する。"""
    provider = AsyncMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(content=content, model="test-model"),
    )
    return provider


def _set_content(provider: AsyncMock, content: str) -> None:
    """モックプロバイダのレスポンス内容を差し替える。"""
    provider.complete.return_value = LLMResponse(content=content, model="test-model")


# --- TestSyntheticDataErrorHierarchy ---


class TestSyntheticDataErrorHierarchy:
    """SyntheticDataError のエラー階層テスト。"""

    def test_inherits_eval_error(self) -> None:
        assert issubclass(SyntheticDataError, EvalError)

    def test_is_exception(self) -> None:
        assert issubclass(SyntheticDataError, Exception)

    def test_can_be_raised_and_caught_as_eval_error(self) -> None:
        with pytest.raises(EvalError):
            raise SyntheticDataError("test")

    def test_message(self) -> None:
        err = SyntheticDataError("合成データ生成に失敗")
        assert str(err) == "合成データ生成に失敗"


# --- TestQAPair ---


class TestQAPair:
    """QAPair モデルのテスト。"""

    def test_creation(self) -> None:
        pair = QAPair(question="質問", answer="回答")
        assert pair.question == "質問"
        assert pair.answer == "回答"

    def test_fields(self) -> None:
        assert set(QAPair.model_fields.keys()) == {"question", "answer"}

    def test_validation_missing_question(self) -> None:
        with pytest.raises(ValueError):
            QAPair(answer="回答")  # type: ignore[call-arg]

    def test_validation_missing_answer(self) -> None:
        with pytest.raises(ValueError):
            QAPair(question="質問")  # type: ignore[call-arg]


# --- TestSyntheticDataGeneratorInit ---


class TestSyntheticDataGeneratorInit:
    """SyntheticDataGenerator のコンストラクタテスト。"""

    def test_default_system_prompt(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        assert gen._system_prompt == SyntheticDataGenerator.DEFAULT_SYSTEM_PROMPT

    def test_custom_system_prompt(self) -> None:
        provider = _make_mock_provider()
        custom = "カスタムプロンプト"
        gen = SyntheticDataGenerator(
            llm_provider=provider, model="test-model", system_prompt=custom
        )
        assert gen._system_prompt == custom

    def test_default_num_pairs(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        assert gen._num_pairs == 3

    def test_custom_num_pairs(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model", num_pairs=5)
        assert gen._num_pairs == 5

    def test_stores_provider_and_model(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="my-model")
        assert gen._provider is provider
        assert gen._model == "my-model"


# --- TestBuildMessages ---


class TestBuildMessages:
    """_build_messages のテスト。"""

    def test_returns_two_messages(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        messages = gen._build_messages("テスト文章", 3)
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        messages = gen._build_messages("テスト文章", 3)
        assert messages[0].role == Role.system
        assert messages[0].content == gen._system_prompt

    def test_second_message_is_user(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        messages = gen._build_messages("テスト文章", 3)
        assert messages[1].role == Role.user

    def test_user_message_contains_text(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        messages = gen._build_messages("これはテスト文章です", 3)
        assert "これはテスト文章です" in messages[1].content

    def test_user_message_contains_num_pairs(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        messages = gen._build_messages("テスト文章", 5)
        assert "5" in messages[1].content

    def test_custom_system_prompt_used(self) -> None:
        provider = _make_mock_provider()
        custom = "カスタムプロンプト"
        gen = SyntheticDataGenerator(
            llm_provider=provider, model="test-model", system_prompt=custom
        )
        messages = gen._build_messages("テスト文章", 3)
        assert messages[0].content == custom


# --- TestParseResponse ---


class TestParseResponse:
    """_parse_response のテスト。"""

    def test_parses_valid_json_array(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = '[{"question": "Q1", "answer": "A1"}]'
        result = gen._parse_response(content)
        assert len(result) == 1
        assert result[0].query == "Q1"
        assert result[0].expected_answer == "A1"

    def test_parses_multiple_items(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]'
        result = gen._parse_response(content)
        assert len(result) == 2
        assert result[0].query == "Q1"
        assert result[1].query == "Q2"

    def test_strips_markdown_code_fence(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = '```json\n[{"question": "Q1", "answer": "A1"}]\n```'
        result = gen._parse_response(content)
        assert len(result) == 1
        assert result[0].query == "Q1"

    def test_strips_code_fence_without_language(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = '```\n[{"question": "Q1", "answer": "A1"}]\n```'
        result = gen._parse_response(content)
        assert len(result) == 1

    def test_skips_invalid_items(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = (
            '[{"question": "Q1", "answer": "A1"}, '
            '{"invalid": true}, '
            '{"question": "Q2", "answer": "A2"}]'
        )
        result = gen._parse_response(content)
        assert len(result) == 2
        assert result[0].query == "Q1"
        assert result[1].query == "Q2"

    def test_skips_items_with_empty_question(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = '[{"question": "", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]'
        result = gen._parse_response(content)
        assert len(result) == 1
        assert result[0].query == "Q2"

    def test_skips_items_with_empty_answer(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = '[{"question": "Q1", "answer": ""}, {"question": "Q2", "answer": "A2"}]'
        result = gen._parse_response(content)
        assert len(result) == 1
        assert result[0].query == "Q2"

    def test_raises_on_empty_array(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError):
            gen._parse_response("[]")

    def test_raises_on_all_invalid_items(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        content = '[{"invalid": true}, {"also": "invalid"}]'
        with pytest.raises(SyntheticDataError):
            gen._parse_response(content)

    def test_raises_on_invalid_json(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError):
            gen._parse_response("not json at all")

    def test_raises_on_json_object_instead_of_array(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError):
            gen._parse_response('{"question": "Q", "answer": "A"}')


# --- TestSyntheticDataGeneratorGenerate ---


class TestSyntheticDataGeneratorGenerate:
    """generate() メソッドのテスト。"""

    @pytest.mark.asyncio
    async def test_returns_eval_dataset(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        result = await gen.generate("テスト文章")
        assert isinstance(result, EvalDataset)

    @pytest.mark.asyncio
    async def test_dataset_name_is_synthetic(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        result = await gen.generate("テスト文章")
        assert result.name == "synthetic"

    @pytest.mark.asyncio
    async def test_dataset_examples_have_correct_fields(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        result = await gen.generate("テスト文章")
        assert len(result.examples) == 1
        ex = result.examples[0]
        assert ex.query == "Q1"
        assert ex.expected_answer == "A1"

    @pytest.mark.asyncio
    async def test_calls_provider_complete(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        await gen.generate("テスト文章")
        provider.complete.assert_awaited_once()
        call_kwargs = provider.complete.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_provider(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        await gen.generate("テスト文章", temperature=0.5)
        call_kwargs = provider.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_num_pairs_override(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model", num_pairs=3)
        await gen.generate("テスト文章", num_pairs=7)
        call_kwargs = provider.complete.call_args
        messages = call_kwargs.kwargs["messages"]
        assert "7" in messages[1].content

    @pytest.mark.asyncio
    async def test_uses_default_num_pairs(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model", num_pairs=4)
        await gen.generate("テスト文章")
        call_kwargs = provider.complete.call_args
        messages = call_kwargs.kwargs["messages"]
        assert "4" in messages[1].content

    @pytest.mark.asyncio
    async def test_raises_on_empty_text(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError):
            await gen.generate("")

    @pytest.mark.asyncio
    async def test_raises_on_whitespace_only_text(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError):
            await gen.generate("   \n\t  ")

    @pytest.mark.asyncio
    async def test_raises_on_llm_failure(self) -> None:
        provider = _make_mock_provider()
        provider.complete.side_effect = RuntimeError("API error")
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError) as exc_info:
            await gen.generate("テスト文章")
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_raises_on_invalid_llm_response(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, "not valid json")
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError):
            await gen.generate("テスト文章")


# --- TestSyntheticDataGeneratorGenerateFromChunks ---


class TestSyntheticDataGeneratorGenerateFromChunks:
    """generate_from_chunks() メソッドのテスト。"""

    @pytest.mark.asyncio
    async def test_combines_results_from_multiple_chunks(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        result = await gen.generate_from_chunks(["chunk1", "chunk2"])
        assert isinstance(result, EvalDataset)
        assert len(result.examples) == 2

    @pytest.mark.asyncio
    async def test_num_pairs_per_chunk(self) -> None:
        provider = _make_mock_provider()
        _set_content(
            provider,
            '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]',
        )
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        result = await gen.generate_from_chunks(["chunk1"], num_pairs_per_chunk=2)
        assert len(result.examples) == 2

    @pytest.mark.asyncio
    async def test_raises_on_empty_chunks(self) -> None:
        provider = _make_mock_provider()
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        with pytest.raises(SyntheticDataError):
            await gen.generate_from_chunks([])

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_provider(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        await gen.generate_from_chunks(["chunk1"], temperature=0.7)
        call_kwargs = provider.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_dataset_name_is_synthetic(self) -> None:
        provider = _make_mock_provider()
        _set_content(provider, '[{"question": "Q1", "answer": "A1"}]')
        gen = SyntheticDataGenerator(llm_provider=provider, model="test-model")
        result = await gen.generate_from_chunks(["chunk1"])
        assert result.name == "synthetic"

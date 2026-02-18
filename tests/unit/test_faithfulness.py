"""Tests for faithfulness metric and parse_evaluation_response helper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.eval import EvalError, MetricError
from src.eval.metrics import parse_evaluation_response
from src.eval.metrics.faithfulness import FaithfulnessMetric, FaithfulnessResult
from src.llm.providers.base import LLMResponse

# =============================================================================
# EvalError / MetricError 例外階層
# =============================================================================


class TestEvalErrorHierarchy:
    """EvalError / MetricError の例外階層テスト。"""

    def test_eval_error_is_exception(self) -> None:
        """EvalError が Exception を継承していること。"""
        assert issubclass(EvalError, Exception)

    def test_metric_error_is_eval_error(self) -> None:
        """MetricError が EvalError を継承していること。"""
        assert issubclass(MetricError, EvalError)

    def test_metric_error_can_be_raised_and_caught_as_eval_error(self) -> None:
        """MetricError を EvalError として捕捉できること。"""
        with pytest.raises(EvalError):
            raise MetricError("test")


# =============================================================================
# parse_evaluation_response
# =============================================================================


class TestParseEvaluationResponse:
    """parse_evaluation_response() のテスト。"""

    def test_parses_score_and_reason(self) -> None:
        """Score と Reason を正しくパースすること。"""
        content = "Score: 0.8\nReason: The answer is faithful."
        score, reason = parse_evaluation_response(content)

        assert score == 0.8
        assert reason == "The answer is faithful."

    def test_parses_integer_score(self) -> None:
        """整数スコア (1, 0) をパースすること。"""
        content = "Score: 1\nReason: Perfect."
        score, reason = parse_evaluation_response(content)

        assert score == 1.0
        assert reason == "Perfect."

    def test_clamps_score_above_one(self) -> None:
        """1.0 を超えるスコアが 1.0 にクランプされること。"""
        content = "Score: 1.5\nReason: Over."
        score, _ = parse_evaluation_response(content)

        assert score == 1.0

    def test_clamps_score_below_zero(self) -> None:
        """0.0 未満のスコアが 0.0 にクランプされること。"""
        content = "Score: -0.3\nReason: Under."
        score, _ = parse_evaluation_response(content)

        assert score == 0.0

    def test_parses_multiline_reason(self) -> None:
        """複数行の理由をパースすること。"""
        content = "Score: 0.5\nReason: Line one.\nLine two."
        _, reason = parse_evaluation_response(content)

        assert "Line one." in reason
        assert "Line two." in reason

    def test_raises_metric_error_on_missing_score(self) -> None:
        """Score がない場合に MetricError が発生すること。"""
        content = "Reason: No score here."

        with pytest.raises(MetricError, match="Score"):
            parse_evaluation_response(content)

    def test_raises_metric_error_on_invalid_score(self) -> None:
        """Score が数値でない場合に MetricError が発生すること。"""
        content = "Score: abc\nReason: Invalid."

        with pytest.raises(MetricError, match="Score"):
            parse_evaluation_response(content)

    def test_returns_empty_reason_when_missing(self) -> None:
        """Reason がない場合に空文字が返ること。"""
        content = "Score: 0.7"
        _, reason = parse_evaluation_response(content)

        assert reason == ""


# =============================================================================
# FaithfulnessResult
# =============================================================================


class TestFaithfulnessResult:
    """FaithfulnessResult モデルのテスト。"""

    def test_creates_with_score_and_reason(self) -> None:
        """score と reason で生成できること。"""
        result = FaithfulnessResult(score=0.9, reason="Faithful answer.")

        assert result.score == 0.9
        assert result.reason == "Faithful answer."

    def test_score_is_float(self) -> None:
        """score が float であること。"""
        result = FaithfulnessResult(score=0.5, reason="test")

        assert isinstance(result.score, float)


# =============================================================================
# FaithfulnessMetric
# =============================================================================


def _make_llm_response(content: str, model: str = "test-model") -> LLMResponse:
    """テスト用の LLMResponse を生成。"""
    return LLMResponse(content=content, model=model)


class TestFaithfulnessMetricEvaluate:
    """FaithfulnessMetric.evaluate() のテスト。"""

    @pytest.fixture
    def provider(self) -> AsyncMock:
        mock = AsyncMock()
        mock.complete.return_value = _make_llm_response(
            "Score: 0.9\nReason: The answer is well supported."
        )
        return mock

    @pytest.fixture
    def metric(self, provider: AsyncMock) -> FaithfulnessMetric:
        return FaithfulnessMetric(llm_provider=provider, model="test-model")

    async def test_returns_faithfulness_result(self, metric: FaithfulnessMetric) -> None:
        """FaithfulnessResult が返ること。"""
        result = await metric.evaluate(context="Some context.", answer="Some answer.")

        assert isinstance(result, FaithfulnessResult)

    async def test_score_between_zero_and_one(self, metric: FaithfulnessMetric) -> None:
        """スコアが 0.0-1.0 の範囲であること。"""
        result = await metric.evaluate(context="ctx", answer="ans")

        assert 0.0 <= result.score <= 1.0

    async def test_high_score_for_faithful_answer(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """忠実な回答に高スコアが返ること (LLMモック)。"""
        provider.complete.return_value = _make_llm_response("Score: 0.95\nReason: Fully faithful.")

        result = await metric.evaluate(context="The sky is blue.", answer="The sky is blue.")

        assert result.score >= 0.9

    async def test_low_score_for_hallucination(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """ハルシネーションに低スコアが返ること (LLMモック)。"""
        provider.complete.return_value = _make_llm_response(
            "Score: 0.1\nReason: Contains hallucination."
        )

        result = await metric.evaluate(
            context="The sky is blue.", answer="The sky is green and purple."
        )

        assert result.score <= 0.3

    async def test_result_contains_reason(self, metric: FaithfulnessMetric) -> None:
        """評価理由が含まれること。"""
        result = await metric.evaluate(context="ctx", answer="ans")

        assert len(result.reason) > 0

    async def test_calls_provider_with_correct_messages(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """provider.complete が正しいメッセージで呼ばれること。"""
        await metric.evaluate(context="Test context.", answer="Test answer.")

        provider.complete.assert_awaited_once()
        call_kwargs = provider.complete.call_args
        messages = call_kwargs.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert "Test context." in messages[1].content
        assert "Test answer." in messages[1].content

    async def test_passes_model_to_provider(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """model が provider.complete に渡されること。"""
        await metric.evaluate(context="ctx", answer="ans")

        call_kwargs = provider.complete.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    async def test_uses_custom_system_prompt(self, provider: AsyncMock) -> None:
        """カスタムシステムプロンプトが使用されること。"""
        custom = "Custom evaluator prompt."
        metric = FaithfulnessMetric(llm_provider=provider, model="test-model", system_prompt=custom)

        await metric.evaluate(context="ctx", answer="ans")

        messages = provider.complete.call_args.kwargs["messages"]
        assert messages[0].content == custom

    async def test_clamps_score_above_one(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """1.0 を超えるスコアが 1.0 にクランプされること。"""
        provider.complete.return_value = _make_llm_response("Score: 1.5\nReason: Over.")

        result = await metric.evaluate(context="ctx", answer="ans")

        assert result.score == 1.0

    async def test_clamps_score_below_zero(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """0.0 未満のスコアが 0.0 にクランプされること。"""
        provider.complete.return_value = _make_llm_response("Score: -0.5\nReason: Under.")

        result = await metric.evaluate(context="ctx", answer="ans")

        assert result.score == 0.0

    async def test_raises_metric_error_on_parse_failure(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """パース失敗時に MetricError が発生すること。"""
        provider.complete.return_value = _make_llm_response("Invalid output")

        with pytest.raises(MetricError):
            await metric.evaluate(context="ctx", answer="ans")

    async def test_raises_metric_error_on_llm_failure(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """LLM エラー時に MetricError が発生すること。"""
        llm_error = RuntimeError("LLM failed")
        provider.complete.side_effect = llm_error

        with pytest.raises(MetricError) as exc_info:
            await metric.evaluate(context="ctx", answer="ans")

        assert exc_info.value.__cause__ is llm_error

    async def test_passes_extra_kwargs_to_provider(
        self, metric: FaithfulnessMetric, provider: AsyncMock
    ) -> None:
        """追加の kwargs が provider.complete に渡されること。"""
        await metric.evaluate(context="ctx", answer="ans", temperature=0.0)

        call_kwargs = provider.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0

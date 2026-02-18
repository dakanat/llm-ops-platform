"""Tests for relevance metric."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.eval import MetricError
from src.eval.metrics.relevance import RelevanceMetric, RelevanceResult
from src.llm.providers.base import LLMResponse


def _make_llm_response(content: str, model: str = "test-model") -> LLMResponse:
    """テスト用の LLMResponse を生成。"""
    return LLMResponse(content=content, model=model)


# =============================================================================
# RelevanceResult
# =============================================================================


class TestRelevanceResult:
    """RelevanceResult モデルのテスト。"""

    def test_creates_with_score_and_reason(self) -> None:
        """score と reason で生成できること。"""
        result = RelevanceResult(score=0.8, reason="Relevant answer.")

        assert result.score == 0.8
        assert result.reason == "Relevant answer."

    def test_score_is_float(self) -> None:
        """score が float であること。"""
        result = RelevanceResult(score=1, reason="test")

        assert isinstance(result.score, float)


# =============================================================================
# RelevanceMetric
# =============================================================================


class TestRelevanceMetricEvaluate:
    """RelevanceMetric.evaluate() のテスト。"""

    @pytest.fixture
    def provider(self) -> AsyncMock:
        mock = AsyncMock()
        mock.complete.return_value = _make_llm_response(
            "Score: 0.85\nReason: The answer is relevant to the query."
        )
        return mock

    @pytest.fixture
    def metric(self, provider: AsyncMock) -> RelevanceMetric:
        return RelevanceMetric(llm_provider=provider, model="test-model")

    async def test_returns_relevance_result(self, metric: RelevanceMetric) -> None:
        """RelevanceResult が返ること。"""
        result = await metric.evaluate(query="What is X?", answer="X is Y.")

        assert isinstance(result, RelevanceResult)

    async def test_score_between_zero_and_one(self, metric: RelevanceMetric) -> None:
        """スコアが 0.0-1.0 の範囲であること。"""
        result = await metric.evaluate(query="q", answer="a")

        assert 0.0 <= result.score <= 1.0

    async def test_high_score_for_relevant_answer(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """関連性の高い回答に高スコアが返ること (LLMモック)。"""
        provider.complete.return_value = _make_llm_response("Score: 0.95\nReason: Highly relevant.")

        result = await metric.evaluate(query="What color is the sky?", answer="The sky is blue.")

        assert result.score >= 0.9

    async def test_low_score_for_irrelevant_answer(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """関連性の低い回答に低スコアが返ること (LLMモック)。"""
        provider.complete.return_value = _make_llm_response(
            "Score: 0.1\nReason: Completely irrelevant."
        )

        result = await metric.evaluate(query="What color is the sky?", answer="Pizza is delicious.")

        assert result.score <= 0.3

    async def test_result_contains_reason(self, metric: RelevanceMetric) -> None:
        """評価理由が含まれること。"""
        result = await metric.evaluate(query="q", answer="a")

        assert len(result.reason) > 0

    async def test_calls_provider_with_correct_messages(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """provider.complete が正しいメッセージで呼ばれること。"""
        await metric.evaluate(query="Test query.", answer="Test answer.")

        provider.complete.assert_awaited_once()
        call_kwargs = provider.complete.call_args
        messages = call_kwargs.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert "Test query." in messages[1].content
        assert "Test answer." in messages[1].content

    async def test_passes_model_to_provider(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """model が provider.complete に渡されること。"""
        await metric.evaluate(query="q", answer="a")

        call_kwargs = provider.complete.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    async def test_uses_custom_system_prompt(self, provider: AsyncMock) -> None:
        """カスタムシステムプロンプトが使用されること。"""
        custom = "Custom relevance prompt."
        metric = RelevanceMetric(llm_provider=provider, model="test-model", system_prompt=custom)

        await metric.evaluate(query="q", answer="a")

        messages = provider.complete.call_args.kwargs["messages"]
        assert messages[0].content == custom

    async def test_clamps_score_above_one(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """1.0 を超えるスコアが 1.0 にクランプされること。"""
        provider.complete.return_value = _make_llm_response("Score: 1.5\nReason: Over.")

        result = await metric.evaluate(query="q", answer="a")

        assert result.score == 1.0

    async def test_clamps_score_below_zero(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """0.0 未満のスコアが 0.0 にクランプされること。"""
        provider.complete.return_value = _make_llm_response("Score: -0.5\nReason: Under.")

        result = await metric.evaluate(query="q", answer="a")

        assert result.score == 0.0

    async def test_raises_metric_error_on_parse_failure(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """パース失敗時に MetricError が発生すること。"""
        provider.complete.return_value = _make_llm_response("Invalid output")

        with pytest.raises(MetricError):
            await metric.evaluate(query="q", answer="a")

    async def test_raises_metric_error_on_llm_failure(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """LLM エラー時に MetricError が発生すること。"""
        llm_error = RuntimeError("LLM failed")
        provider.complete.side_effect = llm_error

        with pytest.raises(MetricError) as exc_info:
            await metric.evaluate(query="q", answer="a")

        assert exc_info.value.__cause__ is llm_error

    async def test_passes_extra_kwargs_to_provider(
        self, metric: RelevanceMetric, provider: AsyncMock
    ) -> None:
        """追加の kwargs が provider.complete に渡されること。"""
        await metric.evaluate(query="q", answer="a", temperature=0.0)

        call_kwargs = provider.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0

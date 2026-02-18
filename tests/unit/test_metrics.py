"""Tests for Prometheus metrics collection (monitoring/metrics.py).

Prometheusメトリクスが正しく記録されること:
- LLMリクエストカウンター (プロバイダ, モデル, ステータス別)
- リクエストレイテンシヒストグラム
- トークン使用量カウンター (prompt / completion)
- エラーカウンター
- 処理中リクエストゲージ
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry
from src.monitoring.metrics import (
    LLMMetrics,
)


@pytest.fixture
def registry() -> CollectorRegistry:
    """各テスト用に独立した Prometheus レジストリを作成する。"""
    return CollectorRegistry()


@pytest.fixture
def metrics(registry: CollectorRegistry) -> LLMMetrics:
    """テスト用の LLMMetrics インスタンスを生成する。"""
    return LLMMetrics(registry=registry)


# =============================================================================
# LLMMetrics 初期化
# =============================================================================


class TestLLMMetricsInit:
    """LLMMetrics の初期化を検証する。"""

    def test_creates_with_registry(self, registry: CollectorRegistry) -> None:
        """レジストリを渡してインスタンスを生成できること。"""
        m = LLMMetrics(registry=registry)
        assert m is not None

    def test_creates_with_default_registry(self) -> None:
        """レジストリを省略してもインスタンスを生成できること。"""
        m = LLMMetrics()
        assert m is not None


# =============================================================================
# リクエストカウンター
# =============================================================================


class TestRequestCounter:
    """LLM リクエストカウンターの記録を検証する。"""

    def test_record_request_increments_counter(self, metrics: LLMMetrics) -> None:
        """record_request() でカウンターが 1 増加すること。"""
        metrics.record_request(provider="openrouter", model="gpt-4", status="success")

        value = metrics.get_request_count(provider="openrouter", model="gpt-4", status="success")
        assert value == 1

    def test_multiple_requests_accumulate(self, metrics: LLMMetrics) -> None:
        """複数回の record_request() が累積すること。"""
        for _ in range(3):
            metrics.record_request(provider="openrouter", model="gpt-4", status="success")

        value = metrics.get_request_count(provider="openrouter", model="gpt-4", status="success")
        assert value == 3

    def test_different_labels_tracked_independently(self, metrics: LLMMetrics) -> None:
        """異なるラベルの組み合わせが独立して追跡されること。"""
        metrics.record_request(provider="openrouter", model="gpt-4", status="success")
        metrics.record_request(provider="openai", model="gpt-4", status="success")

        openrouter_count = metrics.get_request_count(
            provider="openrouter", model="gpt-4", status="success"
        )
        openai_count = metrics.get_request_count(provider="openai", model="gpt-4", status="success")
        assert openrouter_count == 1
        assert openai_count == 1

    def test_error_status_tracked(self, metrics: LLMMetrics) -> None:
        """status='error' のリクエストが追跡されること。"""
        metrics.record_request(provider="openrouter", model="gpt-4", status="error")

        value = metrics.get_request_count(provider="openrouter", model="gpt-4", status="error")
        assert value == 1


# =============================================================================
# レイテンシヒストグラム
# =============================================================================


class TestLatencyHistogram:
    """LLM リクエストレイテンシの記録を検証する。"""

    def test_record_latency_observes_value(self, metrics: LLMMetrics) -> None:
        """record_latency() でヒストグラムに値が記録されること。"""
        metrics.record_latency(provider="openrouter", model="gpt-4", duration_seconds=0.5)

        count = metrics.get_latency_count(provider="openrouter", model="gpt-4")
        assert count == 1

    def test_latency_sum_accumulates(self, metrics: LLMMetrics) -> None:
        """複数回のレイテンシ記録で合計が蓄積されること。"""
        metrics.record_latency(provider="openrouter", model="gpt-4", duration_seconds=1.0)
        metrics.record_latency(provider="openrouter", model="gpt-4", duration_seconds=2.0)

        total = metrics.get_latency_sum(provider="openrouter", model="gpt-4")
        assert total == pytest.approx(3.0)

    def test_latency_count_increments(self, metrics: LLMMetrics) -> None:
        """複数回のレイテンシ記録でカウントが増加すること。"""
        metrics.record_latency(provider="openrouter", model="gpt-4", duration_seconds=0.5)
        metrics.record_latency(provider="openrouter", model="gpt-4", duration_seconds=0.3)

        count = metrics.get_latency_count(provider="openrouter", model="gpt-4")
        assert count == 2


# =============================================================================
# トークンカウンター
# =============================================================================


class TestTokenCounter:
    """トークン使用量カウンターを検証する。"""

    def test_record_tokens_prompt(self, metrics: LLMMetrics) -> None:
        """prompt トークンが記録されること。"""
        metrics.record_tokens(
            provider="openrouter", model="gpt-4", prompt_tokens=100, completion_tokens=50
        )

        prompt_count = metrics.get_token_count(
            provider="openrouter", model="gpt-4", token_type="prompt"
        )
        assert prompt_count == 100

    def test_record_tokens_completion(self, metrics: LLMMetrics) -> None:
        """completion トークンが記録されること。"""
        metrics.record_tokens(
            provider="openrouter", model="gpt-4", prompt_tokens=100, completion_tokens=50
        )

        completion_count = metrics.get_token_count(
            provider="openrouter", model="gpt-4", token_type="completion"
        )
        assert completion_count == 50

    def test_tokens_accumulate(self, metrics: LLMMetrics) -> None:
        """複数回のトークン記録が累積すること。"""
        metrics.record_tokens(
            provider="openrouter", model="gpt-4", prompt_tokens=100, completion_tokens=50
        )
        metrics.record_tokens(
            provider="openrouter", model="gpt-4", prompt_tokens=200, completion_tokens=100
        )

        prompt_total = metrics.get_token_count(
            provider="openrouter", model="gpt-4", token_type="prompt"
        )
        completion_total = metrics.get_token_count(
            provider="openrouter", model="gpt-4", token_type="completion"
        )
        assert prompt_total == 300
        assert completion_total == 150


# =============================================================================
# エラーカウンター
# =============================================================================


class TestErrorCounter:
    """エラーカウンターを検証する。"""

    def test_record_error_increments_counter(self, metrics: LLMMetrics) -> None:
        """record_error() でエラーカウンターが増加すること。"""
        metrics.record_error(provider="openrouter", error_type="timeout")

        value = metrics.get_error_count(provider="openrouter", error_type="timeout")
        assert value == 1

    def test_different_error_types_tracked_independently(self, metrics: LLMMetrics) -> None:
        """異なるエラータイプが独立して追跡されること。"""
        metrics.record_error(provider="openrouter", error_type="timeout")
        metrics.record_error(provider="openrouter", error_type="rate_limit")

        timeout_count = metrics.get_error_count(provider="openrouter", error_type="timeout")
        rate_limit_count = metrics.get_error_count(provider="openrouter", error_type="rate_limit")
        assert timeout_count == 1
        assert rate_limit_count == 1


# =============================================================================
# 処理中リクエストゲージ
# =============================================================================


class TestInProgressGauge:
    """処理中リクエスト数のゲージを検証する。"""

    def test_track_in_progress_increment(self, metrics: LLMMetrics) -> None:
        """track_in_progress() で処理中リクエスト数が増加すること。"""
        metrics.track_in_progress(provider="openrouter", model="gpt-4", delta=1)

        value = metrics.get_in_progress(provider="openrouter", model="gpt-4")
        assert value == 1

    def test_track_in_progress_decrement(self, metrics: LLMMetrics) -> None:
        """track_in_progress() で処理中リクエスト数が減少すること。"""
        metrics.track_in_progress(provider="openrouter", model="gpt-4", delta=1)
        metrics.track_in_progress(provider="openrouter", model="gpt-4", delta=-1)

        value = metrics.get_in_progress(provider="openrouter", model="gpt-4")
        assert value == 0

    def test_in_progress_context_manager(self, metrics: LLMMetrics) -> None:
        """コンテキストマネージャで処理中カウントが管理されること。"""
        with metrics.in_progress(provider="openrouter", model="gpt-4"):
            value = metrics.get_in_progress(provider="openrouter", model="gpt-4")
            assert value == 1

        value_after = metrics.get_in_progress(provider="openrouter", model="gpt-4")
        assert value_after == 0

    def test_in_progress_context_manager_on_exception(self, metrics: LLMMetrics) -> None:
        """例外発生時でもコンテキストマネージャで処理中カウントが減少すること。"""
        with pytest.raises(ValueError), metrics.in_progress(provider="openrouter", model="gpt-4"):
            raise ValueError("test error")

        value = metrics.get_in_progress(provider="openrouter", model="gpt-4")
        assert value == 0

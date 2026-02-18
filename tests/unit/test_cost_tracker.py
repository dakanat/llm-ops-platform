"""Tests for LLM cost tracking and alerts (monitoring/cost_tracker.py).

トークン数からコスト計算、日次コスト追跡、アラート閾値判定を検証する。
"""

from __future__ import annotations

import pytest
from src.llm.providers.base import TokenUsage
from src.monitoring.cost_tracker import CostTracker, ModelPricing

# =============================================================================
# ModelPricing
# =============================================================================


class TestModelPricing:
    """ModelPricing データモデルを検証する。"""

    def test_create_pricing(self) -> None:
        """ModelPricing をインスタンス化できること。"""
        pricing = ModelPricing(
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        )
        assert pricing.input_cost_per_million == 3.0
        assert pricing.output_cost_per_million == 15.0

    def test_zero_cost_pricing(self) -> None:
        """無料モデルのコスト設定を表現できること。"""
        pricing = ModelPricing(
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
        )
        assert pricing.input_cost_per_million == 0.0
        assert pricing.output_cost_per_million == 0.0


# =============================================================================
# CostTracker 初期化
# =============================================================================


class TestCostTrackerInit:
    """CostTracker の初期化を検証する。"""

    def test_creates_with_default_threshold(self) -> None:
        """デフォルトのアラート閾値で生成できること。"""
        tracker = CostTracker()
        assert tracker.alert_threshold_daily_usd == 10.0

    def test_creates_with_custom_threshold(self) -> None:
        """カスタムアラート閾値で生成できること。"""
        tracker = CostTracker(alert_threshold_daily_usd=50.0)
        assert tracker.alert_threshold_daily_usd == 50.0

    def test_initial_daily_cost_is_zero(self) -> None:
        """初期の日次コストが 0 であること。"""
        tracker = CostTracker()
        assert tracker.get_daily_cost() == 0.0

    def test_register_model_pricing(self) -> None:
        """モデルの料金設定を登録できること。"""
        tracker = CostTracker()
        pricing = ModelPricing(input_cost_per_million=3.0, output_cost_per_million=15.0)
        tracker.register_pricing("gpt-4", pricing)

        assert tracker.get_pricing("gpt-4") is pricing

    def test_get_pricing_returns_none_for_unknown_model(self) -> None:
        """未登録モデルの料金取得で None を返すこと。"""
        tracker = CostTracker()
        assert tracker.get_pricing("unknown-model") is None


# =============================================================================
# コスト計算
# =============================================================================


class TestCostCalculation:
    """トークン数からのコスト計算を検証する。"""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        """料金設定済みの CostTracker を生成する。"""
        t = CostTracker()
        t.register_pricing(
            "gpt-4",
            ModelPricing(input_cost_per_million=30.0, output_cost_per_million=60.0),
        )
        t.register_pricing(
            "openai/gpt-oss-120b:free",
            ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0),
        )
        return t

    def test_calculate_cost_basic(self, tracker: CostTracker) -> None:
        """基本的なコスト計算が正しいこと。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        cost = tracker.calculate_cost("gpt-4", usage)

        # input: 1000 / 1_000_000 * 30.0 = 0.03
        # output: 500 / 1_000_000 * 60.0 = 0.03
        # total: 0.06
        assert cost == pytest.approx(0.06)

    def test_calculate_cost_free_model(self, tracker: CostTracker) -> None:
        """無料モデルのコストが 0 であること。"""
        usage = TokenUsage(prompt_tokens=10000, completion_tokens=5000, total_tokens=15000)

        cost = tracker.calculate_cost("openai/gpt-oss-120b:free", usage)

        assert cost == 0.0

    def test_calculate_cost_unknown_model_returns_zero(self, tracker: CostTracker) -> None:
        """未登録モデルのコスト計算が 0 を返すこと。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        cost = tracker.calculate_cost("unknown-model", usage)

        assert cost == 0.0

    def test_calculate_cost_large_token_count(self, tracker: CostTracker) -> None:
        """大量トークンのコスト計算が正しいこと。"""
        usage = TokenUsage(
            prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000
        )

        cost = tracker.calculate_cost("gpt-4", usage)

        # input: 1_000_000 / 1_000_000 * 30.0 = 30.0
        # output: 500_000 / 1_000_000 * 60.0 = 30.0
        # total: 60.0
        assert cost == pytest.approx(60.0)

    def test_calculate_cost_zero_tokens(self, tracker: CostTracker) -> None:
        """トークン数 0 のコスト計算が 0 を返すこと。"""
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        cost = tracker.calculate_cost("gpt-4", usage)

        assert cost == 0.0


# =============================================================================
# 日次コスト追跡
# =============================================================================


class TestDailyCostTracking:
    """日次コスト追跡を検証する。"""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        t = CostTracker()
        t.register_pricing(
            "gpt-4",
            ModelPricing(input_cost_per_million=30.0, output_cost_per_million=60.0),
        )
        return t

    def test_record_cost_updates_daily_total(self, tracker: CostTracker) -> None:
        """record_cost() で日次コストが更新されること。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        tracker.record_cost("gpt-4", usage)

        assert tracker.get_daily_cost() == pytest.approx(0.06)

    def test_multiple_records_accumulate(self, tracker: CostTracker) -> None:
        """複数回の record_cost() がコストを累積すること。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        tracker.record_cost("gpt-4", usage)
        tracker.record_cost("gpt-4", usage)

        assert tracker.get_daily_cost() == pytest.approx(0.12)

    def test_reset_daily_clears_cost(self, tracker: CostTracker) -> None:
        """reset_daily() で日次コストが 0 にリセットされること。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        tracker.record_cost("gpt-4", usage)

        tracker.reset_daily()

        assert tracker.get_daily_cost() == 0.0

    def test_record_cost_returns_cost_value(self, tracker: CostTracker) -> None:
        """record_cost() が記録されたコスト値を返すこと。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        cost = tracker.record_cost("gpt-4", usage)

        assert cost == pytest.approx(0.06)


# =============================================================================
# アラート閾値
# =============================================================================


class TestAlertThreshold:
    """アラート閾値判定を検証する。"""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        t = CostTracker(alert_threshold_daily_usd=1.0)
        t.register_pricing(
            "gpt-4",
            ModelPricing(input_cost_per_million=30.0, output_cost_per_million=60.0),
        )
        return t

    def test_below_threshold_returns_false(self, tracker: CostTracker) -> None:
        """コストが閾値未満の場合に False を返すこと。"""
        assert tracker.is_alert_triggered() is False

    def test_above_threshold_returns_true(self, tracker: CostTracker) -> None:
        """コストが閾値を超えた場合に True を返すこと。"""
        # 0.06 per call → need ~17 calls to exceed $1.0
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        for _ in range(20):
            tracker.record_cost("gpt-4", usage)

        assert tracker.is_alert_triggered() is True

    def test_at_threshold_returns_true(self, tracker: CostTracker) -> None:
        """コストがちょうど閾値に達した場合に True を返すこと。"""
        # Manually set to exactly threshold by recording precise amounts
        tracker = CostTracker(alert_threshold_daily_usd=0.06)
        tracker.register_pricing(
            "gpt-4",
            ModelPricing(input_cost_per_million=30.0, output_cost_per_million=60.0),
        )
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        tracker.record_cost("gpt-4", usage)

        assert tracker.is_alert_triggered() is True

    def test_reset_clears_alert(self, tracker: CostTracker) -> None:
        """reset_daily() 後にアラートがクリアされること。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        for _ in range(20):
            tracker.record_cost("gpt-4", usage)

        tracker.reset_daily()

        assert tracker.is_alert_triggered() is False


# =============================================================================
# コストレポート
# =============================================================================


class TestCostReport:
    """コストレポート生成を検証する。"""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        t = CostTracker()
        t.register_pricing(
            "gpt-4",
            ModelPricing(input_cost_per_million=30.0, output_cost_per_million=60.0),
        )
        t.register_pricing(
            "gpt-3.5-turbo",
            ModelPricing(input_cost_per_million=0.5, output_cost_per_million=1.5),
        )
        return t

    def test_get_cost_report_empty(self, tracker: CostTracker) -> None:
        """コスト記録がない場合のレポートが空であること。"""
        report = tracker.get_cost_report()

        assert report["total_cost"] == 0.0
        assert report["model_costs"] == {}

    def test_get_cost_report_single_model(self, tracker: CostTracker) -> None:
        """単一モデルのコストレポートが正しいこと。"""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        tracker.record_cost("gpt-4", usage)

        report = tracker.get_cost_report()

        assert report["total_cost"] == pytest.approx(0.06)
        assert "gpt-4" in report["model_costs"]
        assert report["model_costs"]["gpt-4"]["cost"] == pytest.approx(0.06)
        assert report["model_costs"]["gpt-4"]["requests"] == 1

    def test_get_cost_report_multiple_models(self, tracker: CostTracker) -> None:
        """複数モデルのコストレポートが正しいこと。"""
        usage_gpt4 = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        usage_gpt35 = TokenUsage(prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)

        tracker.record_cost("gpt-4", usage_gpt4)
        tracker.record_cost("gpt-3.5-turbo", usage_gpt35)

        report = tracker.get_cost_report()

        assert "gpt-4" in report["model_costs"]
        assert "gpt-3.5-turbo" in report["model_costs"]
        assert report["model_costs"]["gpt-4"]["requests"] == 1
        assert report["model_costs"]["gpt-3.5-turbo"]["requests"] == 1
        # Total = gpt4 cost + gpt3.5 cost
        gpt4_cost = report["model_costs"]["gpt-4"]["cost"]
        gpt35_cost = report["model_costs"]["gpt-3.5-turbo"]["cost"]
        assert report["total_cost"] == pytest.approx(gpt4_cost + gpt35_cost)

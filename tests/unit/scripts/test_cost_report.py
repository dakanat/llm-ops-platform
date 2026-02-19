"""Tests for scripts/cost_report.py — cost report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.cost_report import (
    PricingEntry,
    UsageData,
    UsageRecord,
    build_arg_parser,
    format_cost_report,
    generate_report,
    load_usage_data,
)
from src.monitoring.cost_tracker import CostReport, ModelCostSummary


class TestUsageDataModels:
    """Pydantic モデルの生成・バリデーション検証。"""

    def test_usage_record_creation(self) -> None:
        """UsageRecord を正しく生成できる。"""
        record = UsageRecord(model="gpt-4", prompt_tokens=1000, completion_tokens=500)
        assert record.model == "gpt-4"
        assert record.prompt_tokens == 1000
        assert record.completion_tokens == 500

    def test_pricing_entry_creation(self) -> None:
        """PricingEntry を正しく生成できる。"""
        entry = PricingEntry(input_cost_per_million=30.0, output_cost_per_million=60.0)
        assert entry.input_cost_per_million == 30.0
        assert entry.output_cost_per_million == 60.0

    def test_usage_data_creation(self) -> None:
        """UsageData を正しく生成できる。"""
        data = UsageData(
            records=[
                UsageRecord(model="gpt-4", prompt_tokens=1000, completion_tokens=500),
            ],
            pricing={
                "gpt-4": PricingEntry(input_cost_per_million=30.0, output_cost_per_million=60.0),
            },
        )
        assert len(data.records) == 1
        assert "gpt-4" in data.pricing

    def test_usage_data_empty_records(self) -> None:
        """空のレコードリストでも生成可能。"""
        data = UsageData(records=[], pricing={})
        assert len(data.records) == 0


class TestBuildArgParser:
    """CLI引数パーサの検証。"""

    def test_no_args_required(self) -> None:
        """引数なしでもパースできる。"""
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.usage_file is None

    def test_usage_file_argument(self) -> None:
        """--usage-file が正しくパースされる。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--usage-file", "usage.json"])
        assert args.usage_file == "usage.json"

    def test_threshold_default(self) -> None:
        """--threshold のデフォルトは None (Settings値を使用)。"""
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.threshold is None

    def test_threshold_argument(self) -> None:
        """--threshold が正しくパースされる。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--threshold", "20.0"])
        assert args.threshold == 20.0


class TestLoadUsageData:
    """使用量データのJSON読み込み検証。"""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """正常なJSONを正しく読み込む。"""
        data = {
            "records": [{"model": "gpt-4", "prompt_tokens": 1000, "completion_tokens": 500}],
            "pricing": {"gpt-4": {"input_cost_per_million": 30.0, "output_cost_per_million": 60.0}},
        }
        file_path = tmp_path / "usage.json"
        file_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_usage_data(file_path)
        assert len(result.records) == 1
        assert result.records[0].model == "gpt-4"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """ファイルが存在しない場合に FileNotFoundError を送出する。"""
        with pytest.raises(FileNotFoundError):
            load_usage_data(tmp_path / "missing.json")

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """不正なJSONで ValueError を送出する。"""
        file_path = tmp_path / "bad.json"
        file_path.write_text("not-valid-json", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON"):
            load_usage_data(file_path)

    def test_raises_on_validation_error(self, tmp_path: Path) -> None:
        """バリデーションエラーで ValueError を送出する。"""
        file_path = tmp_path / "invalid.json"
        file_path.write_text(json.dumps({"records": "not-a-list"}), encoding="utf-8")

        with pytest.raises(ValueError, match="バリデーション"):
            load_usage_data(file_path)


class TestGenerateReport:
    """レポート生成の検証。"""

    def test_single_model_cost(self) -> None:
        """単一モデルのコスト計算が正しい。"""
        data = UsageData(
            records=[
                UsageRecord(model="gpt-4", prompt_tokens=1_000_000, completion_tokens=500_000),
            ],
            pricing={
                "gpt-4": PricingEntry(input_cost_per_million=30.0, output_cost_per_million=60.0),
            },
        )

        report, alert = generate_report(data, alert_threshold=100.0)
        # input: 1M tokens * 30.0/M = 30.0
        # output: 500K tokens * 60.0/M = 30.0
        # total = 60.0
        assert report["total_cost"] == pytest.approx(60.0)

    def test_multiple_models(self) -> None:
        """複数モデルのコスト合計が正しい。"""
        data = UsageData(
            records=[
                UsageRecord(model="gpt-4", prompt_tokens=1_000_000, completion_tokens=0),
                UsageRecord(model="gpt-3.5", prompt_tokens=1_000_000, completion_tokens=0),
            ],
            pricing={
                "gpt-4": PricingEntry(input_cost_per_million=30.0, output_cost_per_million=60.0),
                "gpt-3.5": PricingEntry(input_cost_per_million=0.5, output_cost_per_million=1.5),
            },
        )

        report, _ = generate_report(data, alert_threshold=100.0)
        assert report["total_cost"] == pytest.approx(30.5)
        assert "gpt-4" in report["model_costs"]
        assert "gpt-3.5" in report["model_costs"]

    def test_alert_triggered(self) -> None:
        """閾値を超えた場合にアラートが True になる。"""
        data = UsageData(
            records=[
                UsageRecord(model="gpt-4", prompt_tokens=1_000_000, completion_tokens=500_000),
            ],
            pricing={
                "gpt-4": PricingEntry(input_cost_per_million=30.0, output_cost_per_million=60.0),
            },
        )

        _, alert = generate_report(data, alert_threshold=10.0)
        assert alert is True

    def test_alert_not_triggered(self) -> None:
        """閾値以下ならアラートが False になる。"""
        data = UsageData(
            records=[
                UsageRecord(model="gpt-4", prompt_tokens=1000, completion_tokens=500),
            ],
            pricing={
                "gpt-4": PricingEntry(input_cost_per_million=30.0, output_cost_per_million=60.0),
            },
        )

        _, alert = generate_report(data, alert_threshold=100.0)
        assert alert is False

    def test_empty_records(self) -> None:
        """レコードが空の場合のコストは 0。"""
        data = UsageData(records=[], pricing={})
        report, alert = generate_report(data, alert_threshold=10.0)
        assert report["total_cost"] == 0.0
        assert alert is False


class TestFormatCostReport:
    """レポートフォーマットの検証。"""

    def test_includes_total_cost(self) -> None:
        """合計コストが含まれる。"""
        report: CostReport = CostReport(total_cost=42.5, model_costs={})
        output = format_cost_report(report, alert_triggered=False, threshold=100.0)
        assert "42.50" in output

    def test_includes_model_breakdown(self) -> None:
        """モデル別内訳が含まれる。"""
        report: CostReport = CostReport(
            total_cost=30.0,
            model_costs={
                "gpt-4": ModelCostSummary(cost=30.0, requests=5),
            },
        )
        output = format_cost_report(report, alert_triggered=False, threshold=100.0)
        assert "gpt-4" in output
        assert "30.00" in output
        assert "5" in output

    def test_includes_alert_warning(self) -> None:
        """アラート発生時に警告メッセージが含まれる。"""
        report: CostReport = CostReport(total_cost=50.0, model_costs={})
        output = format_cost_report(report, alert_triggered=True, threshold=10.0)
        assert "ALERT" in output

    def test_no_alert_when_below_threshold(self) -> None:
        """アラートなしの場合に警告が含まれない。"""
        report: CostReport = CostReport(total_cost=5.0, model_costs={})
        output = format_cost_report(report, alert_triggered=False, threshold=100.0)
        assert "ALERT" not in output

"""コストレポート生成スクリプト。

使用量データJSONを読み込み、CostTrackerでコスト計算してレポートを生成・表示する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError
from src.config import Settings
from src.llm.providers.base import TokenUsage
from src.monitoring.cost_tracker import CostReport, CostTracker, ModelPricing
from src.monitoring.logger import get_logger, setup_logging


class UsageRecord(BaseModel):
    """1リクエスト分の使用量レコード。

    Attributes:
        model: モデル名。
        prompt_tokens: プロンプトトークン数。
        completion_tokens: コンプリーショントークン数。
    """

    model: str
    prompt_tokens: int
    completion_tokens: int


class PricingEntry(BaseModel):
    """モデルの料金定義。

    Attributes:
        input_cost_per_million: 入力100万トークンあたりのコスト (USD)。
        output_cost_per_million: 出力100万トークンあたりのコスト (USD)。
    """

    input_cost_per_million: float
    output_cost_per_million: float


class UsageData(BaseModel):
    """使用量データ全体。

    Attributes:
        records: 使用量レコードのリスト。
        pricing: モデル名→料金定義のマッピング。
    """

    records: list[UsageRecord]
    pricing: dict[str, PricingEntry]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI引数パーサを構築する。

    Returns:
        設定済みの ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="コストレポート生成")
    parser.add_argument("--usage-file", default=None, help="使用量データJSONパス")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="日次アラート閾値 (USD)。未指定時はSettings値を使用",
    )
    return parser


def load_usage_data(path: Path) -> UsageData:
    """JSONファイルから使用量データを読み込む。

    Args:
        path: JSONファイルのパス。

    Returns:
        読み込んだ UsageData。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        ValueError: JSONパースまたはバリデーションに失敗した場合。
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"JSONのパースに失敗しました: {path}"
        raise ValueError(msg) from e

    try:
        return UsageData.model_validate(data)
    except ValidationError as e:
        msg = f"バリデーションに失敗しました: {e}"
        raise ValueError(msg) from e


def generate_report(
    usage_data: UsageData,
    alert_threshold: float,
) -> tuple[CostReport, bool]:
    """使用量データからコストレポートを生成する。

    Args:
        usage_data: 使用量データ。
        alert_threshold: 日次アラート閾値 (USD)。

    Returns:
        (コストレポート, アラート発生有無) のタプル。
    """
    tracker = CostTracker(alert_threshold_daily_usd=alert_threshold)

    for model, pricing_entry in usage_data.pricing.items():
        tracker.register_pricing(
            model,
            ModelPricing(
                input_cost_per_million=pricing_entry.input_cost_per_million,
                output_cost_per_million=pricing_entry.output_cost_per_million,
            ),
        )

    for record in usage_data.records:
        usage = TokenUsage(
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.prompt_tokens + record.completion_tokens,
        )
        tracker.record_cost(record.model, usage)

    return tracker.get_cost_report(), tracker.is_alert_triggered()


def format_cost_report(
    report: CostReport,
    alert_triggered: bool,
    threshold: float,
) -> str:
    """コストレポートの人間可読な出力を生成する。

    Args:
        report: コストレポート。
        alert_triggered: アラートが発生したかどうか。
        threshold: アラート閾値 (USD)。

    Returns:
        フォーマット済みレポート文字列。
    """
    lines = [
        "=== Cost Report ===",
        f"  Total Cost: ${report['total_cost']:.2f}",
    ]

    if report["model_costs"]:
        lines.append("  Model Breakdown:")
        for model, summary in report["model_costs"].items():
            lines.append(f"    {model}: ${summary['cost']:.2f} ({summary['requests']} requests)")

    if alert_triggered:
        lines.append(
            f"  [ALERT] Daily cost ${report['total_cost']:.2f} exceeds threshold ${threshold:.2f}"
        )

    return "\n".join(lines)


def main() -> None:
    """エントリポイント。CLI引数をパースしてレポートを生成する。"""
    settings = Settings()
    setup_logging(settings.log_level)
    logger = get_logger(script="cost_report")

    parser = build_arg_parser()
    args = parser.parse_args()

    threshold = (
        args.threshold
        if args.threshold is not None
        else float(settings.cost_alert_threshold_daily_usd)
    )

    if args.usage_file is not None:
        usage_data = load_usage_data(Path(args.usage_file))
    else:
        logger.info(
            "no_usage_file", msg="使用量ファイルが指定されていません。空のレポートを生成します。"
        )
        usage_data = UsageData(records=[], pricing={})

    report, alert_triggered = generate_report(usage_data, alert_threshold=threshold)
    output = format_cost_report(report, alert_triggered=alert_triggered, threshold=threshold)
    print(output)

    if alert_triggered:
        logger.warning("cost_alert", total_cost=report["total_cost"], threshold=threshold)


if __name__ == "__main__":
    main()

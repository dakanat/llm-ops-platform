"""評価一括実行CLI。

データセットJSONを読み込み、EvalRunnerで評価実行する。
オプションでベースラインとの回帰比較を行う。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.config import Settings
from src.eval.datasets import load_dataset
from src.eval.metrics.faithfulness import FaithfulnessMetric
from src.eval.metrics.relevance import RelevanceMetric
from src.eval.regression import (
    RegressionResult,
    RegressionThresholds,
    compare,
    load_baseline,
    save_baseline,
)
from src.eval.runner import EvalRunner, EvalRunResult
from src.llm.router import LLMRouter
from src.monitoring.logger import get_logger, setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI引数パーサを構築する。

    Returns:
        設定済みの ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="評価一括実行CLI")
    parser.add_argument("--dataset", required=True, help="評価データセットJSONパス")
    parser.add_argument("--baseline", default=None, help="ベースラインJSONパス (回帰テスト用)")
    parser.add_argument("--output", default=None, help="結果出力先JSONパス")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        default=False,
        help="結果をベースラインとして保存",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.05,
        help="忠実性の回帰閾値 (default: 0.05)",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.05,
        help="関連性の回帰閾値 (default: 0.05)",
    )
    return parser


def format_eval_summary(result: EvalRunResult) -> str:
    """評価結果の人間可読なサマリ文字列を生成する。

    Args:
        result: 評価結果。

    Returns:
        フォーマット済みサマリ文字列。
    """
    lines = [
        f"=== Evaluation Summary: {result.dataset_name} ===",
    ]

    if result.faithfulness_summary is not None:
        lines.append(
            f"  Faithfulness: {result.faithfulness_summary.mean:.4f} "
            f"({result.faithfulness_summary.count} examples)"
        )
    else:
        lines.append("  Faithfulness: N/A")

    if result.relevance_summary is not None:
        lines.append(
            f"  Relevance:    {result.relevance_summary.mean:.4f} "
            f"({result.relevance_summary.count} examples)"
        )
    else:
        lines.append("  Relevance:    N/A")

    return "\n".join(lines)


def format_regression_summary(reg_result: RegressionResult) -> str:
    """回帰テスト結果の人間可読なサマリ文字列を生成する。

    Args:
        reg_result: 回帰テスト結果。

    Returns:
        フォーマット済みサマリ文字列。
    """
    status = "PASSED" if reg_result.passed else "FAILED"
    lines = [
        f"=== Regression Test: {status} ===",
    ]
    for detail in reg_result.details:
        lines.append(f"  {detail}")
    return "\n".join(lines)


async def run_evaluation(
    dataset_path: Path,
    runner: EvalRunner,
    baseline_path: Path | None = None,
    output_path: Path | None = None,
    update_baseline: bool = False,
    faithfulness_threshold: float = 0.05,
    relevance_threshold: float = 0.05,
) -> int:
    """評価を実行し、結果を処理する。

    Args:
        dataset_path: データセットJSONパス。
        runner: 評価実行エンジン。
        baseline_path: ベースラインJSONパス。
        output_path: 結果出力先JSONパス。
        update_baseline: ベースラインを更新するかどうか。
        faithfulness_threshold: 忠実性の回帰閾値。
        relevance_threshold: 関連性の回帰閾値。

    Returns:
        exit code (0=成功, 1=回帰検出)。
    """
    logger = get_logger(script="run_eval")

    dataset = load_dataset(dataset_path)
    logger.info("dataset_loaded", name=dataset.name, examples=len(dataset.examples))

    result = await runner.run(dataset)
    logger.info("evaluation_complete", dataset=result.dataset_name)
    print(format_eval_summary(result))

    if output_path is not None:
        save_baseline(result, output_path)
        logger.info("output_saved", path=str(output_path))

    if update_baseline and baseline_path is not None:
        save_baseline(result, baseline_path)
        logger.info("baseline_updated", path=str(baseline_path))

    if baseline_path is not None and not update_baseline:
        baseline = load_baseline(baseline_path)
        thresholds = RegressionThresholds(
            faithfulness_drop=faithfulness_threshold,
            relevance_drop=relevance_threshold,
        )
        reg_result = compare(result, baseline, thresholds=thresholds)
        print(format_regression_summary(reg_result))

        if not reg_result.passed:
            logger.warning("regression_detected")
            return 1

    return 0


async def main() -> int:
    """エントリポイント。CLI引数をパースして評価を実行する。

    Returns:
        exit code。
    """
    settings = Settings()
    setup_logging(settings.log_level)

    parser = build_arg_parser()
    args = parser.parse_args()

    router = LLMRouter(settings)
    provider = router.get_provider()
    model = router.model

    faithfulness_metric = FaithfulnessMetric(llm_provider=provider, model=model)
    relevance_metric = RelevanceMetric(llm_provider=provider, model=model)
    runner = EvalRunner(
        faithfulness_metric=faithfulness_metric,
        relevance_metric=relevance_metric,
    )

    return await run_evaluation(
        dataset_path=Path(args.dataset),
        runner=runner,
        baseline_path=Path(args.baseline) if args.baseline else None,
        output_path=Path(args.output) if args.output else None,
        update_baseline=args.update_baseline,
        faithfulness_threshold=args.faithfulness_threshold,
        relevance_threshold=args.relevance_threshold,
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

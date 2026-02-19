"""Tests for scripts/run_eval.py — evaluation CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from scripts.run_eval import (
    build_arg_parser,
    format_eval_summary,
    format_regression_summary,
    run_evaluation,
)
from src.eval.regression import RegressionResult
from src.eval.runner import EvalRunResult, MetricSummary


class TestBuildArgParser:
    """CLI引数パーサの検証。"""

    def test_dataset_is_required(self) -> None:
        """--dataset は必須引数。"""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_dataset_argument(self) -> None:
        """--dataset が正しくパースされる。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json"])
        assert args.dataset == "data.json"

    def test_baseline_optional(self) -> None:
        """--baseline はオプション。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json"])
        assert args.baseline is None

    def test_baseline_argument(self) -> None:
        """--baseline が正しくパースされる。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json", "--baseline", "base.json"])
        assert args.baseline == "base.json"

    def test_output_optional(self) -> None:
        """--output はオプション。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json"])
        assert args.output is None

    def test_output_argument(self) -> None:
        """--output が正しくパースされる。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json", "--output", "out.json"])
        assert args.output == "out.json"

    def test_update_baseline_flag_default(self) -> None:
        """--update-baseline のデフォルトは False。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json"])
        assert args.update_baseline is False

    def test_update_baseline_flag(self) -> None:
        """--update-baseline フラグが正しくパースされる。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json", "--update-baseline"])
        assert args.update_baseline is True

    def test_faithfulness_threshold_default(self) -> None:
        """--faithfulness-threshold のデフォルトは 0.05。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json"])
        assert args.faithfulness_threshold == 0.05

    def test_relevance_threshold_default(self) -> None:
        """--relevance-threshold のデフォルトは 0.05。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "data.json"])
        assert args.relevance_threshold == 0.05

    def test_custom_thresholds(self) -> None:
        """カスタム閾値が正しくパースされる。"""
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "data.json",
                "--faithfulness-threshold",
                "0.1",
                "--relevance-threshold",
                "0.2",
            ]
        )
        assert args.faithfulness_threshold == 0.1
        assert args.relevance_threshold == 0.2


class TestFormatEvalSummary:
    """評価サマリのフォーマット検証。"""

    def test_includes_dataset_name(self) -> None:
        """サマリにデータセット名が含まれる。"""
        result = EvalRunResult(dataset_name="test-set", results=[])
        summary = format_eval_summary(result)
        assert "test-set" in summary

    def test_includes_faithfulness_mean(self) -> None:
        """忠実性の平均スコアが含まれる。"""
        result = EvalRunResult(
            dataset_name="test-set",
            results=[],
            faithfulness_summary=MetricSummary(mean=0.85, count=10),
        )
        summary = format_eval_summary(result)
        assert "0.8500" in summary

    def test_includes_relevance_mean(self) -> None:
        """関連性の平均スコアが含まれる。"""
        result = EvalRunResult(
            dataset_name="test-set",
            results=[],
            relevance_summary=MetricSummary(mean=0.92, count=10),
        )
        summary = format_eval_summary(result)
        assert "0.9200" in summary

    def test_handles_no_summaries(self) -> None:
        """サマリなしでもエラーにならない。"""
        result = EvalRunResult(dataset_name="test-set", results=[])
        summary = format_eval_summary(result)
        assert "N/A" in summary

    def test_includes_example_count(self) -> None:
        """サンプル数が含まれる。"""
        result = EvalRunResult(
            dataset_name="test-set",
            results=[],
            faithfulness_summary=MetricSummary(mean=0.85, count=10),
        )
        summary = format_eval_summary(result)
        assert "10" in summary


class TestFormatRegressionSummary:
    """回帰結果のフォーマット検証。"""

    def test_passed_result(self) -> None:
        """合格結果に PASSED が含まれる。"""
        reg = RegressionResult(
            passed=True,
            details=["faithfulness: PASS"],
        )
        summary = format_regression_summary(reg)
        assert "PASSED" in summary

    def test_failed_result(self) -> None:
        """不合格結果に FAILED が含まれる。"""
        reg = RegressionResult(
            passed=False,
            details=["faithfulness: FAIL"],
        )
        summary = format_regression_summary(reg)
        assert "FAILED" in summary

    def test_includes_details(self) -> None:
        """詳細情報が含まれる。"""
        reg = RegressionResult(
            passed=True,
            details=["faithfulness: PASS (baseline=0.90, current=0.88)"],
        )
        summary = format_regression_summary(reg)
        assert "faithfulness: PASS" in summary


class TestRunEvaluation:
    """評価実行フローの検証。"""

    @pytest.mark.asyncio
    async def test_runs_evaluation_and_returns_zero(self) -> None:
        """正常な評価実行で exit code 0 を返す。"""
        mock_dataset = AsyncMock()
        mock_result = EvalRunResult(dataset_name="test", results=[])
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with patch("scripts.run_eval.load_dataset", return_value=mock_dataset):
            exit_code = await run_evaluation(
                dataset_path=Path("data.json"),
                runner=mock_runner,
            )

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_saves_output_when_specified(self) -> None:
        """--output 指定時に結果がファイルに保存される。"""
        mock_dataset = AsyncMock()
        mock_result = EvalRunResult(dataset_name="test", results=[])
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch("scripts.run_eval.load_dataset", return_value=mock_dataset),
            patch("scripts.run_eval.save_baseline") as mock_save,
        ):
            await run_evaluation(
                dataset_path=Path("data.json"),
                runner=mock_runner,
                output_path=Path("out.json"),
            )

        mock_save.assert_called_once_with(mock_result, Path("out.json"))

    @pytest.mark.asyncio
    async def test_regression_pass_returns_zero(self) -> None:
        """回帰テストが合格の場合 exit code 0 を返す。"""
        mock_dataset = AsyncMock()
        mock_result = EvalRunResult(dataset_name="test", results=[])
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        mock_baseline = EvalRunResult(dataset_name="test", results=[])
        mock_reg_result = RegressionResult(passed=True, details=["PASS"])

        with (
            patch("scripts.run_eval.load_dataset", return_value=mock_dataset),
            patch("scripts.run_eval.load_baseline", return_value=mock_baseline),
            patch("scripts.run_eval.compare", return_value=mock_reg_result),
        ):
            exit_code = await run_evaluation(
                dataset_path=Path("data.json"),
                runner=mock_runner,
                baseline_path=Path("base.json"),
            )

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_regression_fail_returns_one(self) -> None:
        """回帰テストが不合格の場合 exit code 1 を返す。"""
        mock_dataset = AsyncMock()
        mock_result = EvalRunResult(dataset_name="test", results=[])
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        mock_baseline = EvalRunResult(dataset_name="test", results=[])
        mock_reg_result = RegressionResult(passed=False, details=["FAIL"])

        with (
            patch("scripts.run_eval.load_dataset", return_value=mock_dataset),
            patch("scripts.run_eval.load_baseline", return_value=mock_baseline),
            patch("scripts.run_eval.compare", return_value=mock_reg_result),
        ):
            exit_code = await run_evaluation(
                dataset_path=Path("data.json"),
                runner=mock_runner,
                baseline_path=Path("base.json"),
            )

        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_update_baseline_saves_result(self) -> None:
        """--update-baseline 指定時にベースラインが保存される。"""
        mock_dataset = AsyncMock()
        mock_result = EvalRunResult(dataset_name="test", results=[])
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        with (
            patch("scripts.run_eval.load_dataset", return_value=mock_dataset),
            patch("scripts.run_eval.save_baseline") as mock_save,
        ):
            await run_evaluation(
                dataset_path=Path("data.json"),
                runner=mock_runner,
                update_baseline=True,
                baseline_path=Path("base.json"),
            )

        mock_save.assert_called_once_with(mock_result, Path("base.json"))

    @pytest.mark.asyncio
    async def test_custom_thresholds_passed_to_compare(self) -> None:
        """カスタム閾値が compare に渡される。"""
        mock_dataset = AsyncMock()
        mock_result = EvalRunResult(dataset_name="test", results=[])
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result

        mock_baseline = EvalRunResult(dataset_name="test", results=[])
        mock_reg_result = RegressionResult(passed=True, details=["PASS"])

        with (
            patch("scripts.run_eval.load_dataset", return_value=mock_dataset),
            patch("scripts.run_eval.load_baseline", return_value=mock_baseline),
            patch("scripts.run_eval.compare", return_value=mock_reg_result) as mock_compare,
        ):
            await run_evaluation(
                dataset_path=Path("data.json"),
                runner=mock_runner,
                baseline_path=Path("base.json"),
                faithfulness_threshold=0.1,
                relevance_threshold=0.2,
            )

        thresholds = mock_compare.call_args.kwargs["thresholds"]
        assert thresholds.faithfulness_drop == 0.1
        assert thresholds.relevance_drop == 0.2

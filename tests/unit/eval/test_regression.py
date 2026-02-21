"""Tests for eval regression testing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.eval import EvalError, RegressionError
from src.eval.regression import (
    RegressionResult,
    RegressionThresholds,
    compare,
    load_baseline,
    save_baseline,
)
from src.eval.runner import EvalRunResult, ExampleResult, MetricSummary

# =============================================================================
# RegressionError 例外階層
# =============================================================================


class TestRegressionError:
    """RegressionError の例外階層テスト。"""

    def test_regression_error_is_eval_error(self) -> None:
        """RegressionError が EvalError を継承していること。"""
        assert issubclass(RegressionError, EvalError)

    def test_regression_error_can_be_raised_and_caught_as_eval_error(self) -> None:
        """RegressionError を EvalError として捕捉できること。"""
        with pytest.raises(EvalError):
            raise RegressionError("test")


# =============================================================================
# RegressionThresholds モデル
# =============================================================================


class TestRegressionThresholds:
    """RegressionThresholds モデルのテスト。"""

    def test_default_values(self) -> None:
        """デフォルト値が設定されること。"""
        thresholds = RegressionThresholds()

        assert thresholds.faithfulness_drop == 0.05
        assert thresholds.relevance_drop == 0.05

    def test_custom_values(self) -> None:
        """カスタム値を指定できること。"""
        thresholds = RegressionThresholds(faithfulness_drop=0.1, relevance_drop=0.2)

        assert thresholds.faithfulness_drop == 0.1
        assert thresholds.relevance_drop == 0.2


# =============================================================================
# RegressionResult モデル
# =============================================================================


class TestRegressionResult:
    """RegressionResult モデルのテスト。"""

    def test_creates_with_required_fields(self) -> None:
        """必須フィールドで生成できること。"""
        result = RegressionResult(passed=True, details=["All metrics passed."])

        assert result.passed is True
        assert result.details == ["All metrics passed."]

    def test_score_fields_are_optional(self) -> None:
        """スコアフィールドがオプションであること。"""
        result = RegressionResult(passed=True, details=[])

        assert result.current_faithfulness is None
        assert result.baseline_faithfulness is None
        assert result.current_relevance is None
        assert result.baseline_relevance is None

    def test_creates_with_all_fields(self) -> None:
        """全フィールドで生成できること。"""
        result = RegressionResult(
            passed=False,
            details=["faithfulness dropped"],
            current_faithfulness=0.7,
            baseline_faithfulness=0.9,
            current_relevance=0.8,
            baseline_relevance=0.85,
        )

        assert result.passed is False
        assert result.current_faithfulness == 0.7
        assert result.baseline_faithfulness == 0.9


# =============================================================================
# load_baseline / save_baseline
# =============================================================================


def _make_eval_run_result(
    faithfulness_mean: float | None = None,
    relevance_mean: float | None = None,
) -> EvalRunResult:
    """テスト用の EvalRunResult を生成。"""
    return EvalRunResult(
        dataset_name="test",
        results=[],
        faithfulness_summary=(
            MetricSummary(mean=faithfulness_mean, count=10)
            if faithfulness_mean is not None
            else None
        ),
        relevance_summary=(
            MetricSummary(mean=relevance_mean, count=10) if relevance_mean is not None else None
        ),
    )


class TestLoadBaseline:
    """load_baseline() のテスト。"""

    def test_loads_valid_baseline(self, tmp_path: Path) -> None:
        """正常なベースラインを読み込めること。"""
        result = _make_eval_run_result(faithfulness_mean=0.9)
        path = tmp_path / "baseline.json"
        path.write_text(result.model_dump_json(), encoding="utf-8")

        loaded = load_baseline(path)

        assert loaded.dataset_name == "test"
        assert loaded.faithfulness_summary is not None
        assert loaded.faithfulness_summary.mean == 0.9

    def test_raises_regression_error_on_file_not_found(self, tmp_path: Path) -> None:
        """存在しないファイルで RegressionError が発生すること。"""
        path = tmp_path / "nonexistent.json"

        with pytest.raises(RegressionError):
            load_baseline(path)

    def test_raises_regression_error_on_invalid_json(self, tmp_path: Path) -> None:
        """不正な JSON で RegressionError が発生すること。"""
        path = tmp_path / "bad.json"
        path.write_text("{invalid", encoding="utf-8")

        with pytest.raises(RegressionError):
            load_baseline(path)

    def test_raises_regression_error_on_validation_error(self, tmp_path: Path) -> None:
        """バリデーションエラーで RegressionError が発生すること。"""
        path = tmp_path / "invalid.json"
        path.write_text(json.dumps({"bad_field": True}), encoding="utf-8")

        with pytest.raises(RegressionError):
            load_baseline(path)


class TestSaveBaseline:
    """save_baseline() のテスト。"""

    def test_saves_to_file(self, tmp_path: Path) -> None:
        """ファイルに書き込めること。"""
        result = _make_eval_run_result(faithfulness_mean=0.9)
        path = tmp_path / "baseline.json"

        save_baseline(result, path)

        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        """save → load のラウンドトリップが成功すること。"""
        original = _make_eval_run_result(faithfulness_mean=0.85, relevance_mean=0.9)
        path = tmp_path / "roundtrip.json"

        save_baseline(original, path)
        loaded = load_baseline(path)

        assert loaded.dataset_name == original.dataset_name
        assert loaded.faithfulness_summary is not None
        assert loaded.faithfulness_summary.mean == 0.85
        assert loaded.relevance_summary is not None
        assert loaded.relevance_summary.mean == 0.9

    def test_roundtrip_with_example_results(self, tmp_path: Path) -> None:
        """ExampleResult を含む結果のラウンドトリップが成功すること。"""
        original = EvalRunResult(
            dataset_name="test",
            results=[ExampleResult(query="q", faithfulness_score=0.8)],
            faithfulness_summary=MetricSummary(mean=0.8, count=1),
        )
        path = tmp_path / "roundtrip.json"

        save_baseline(original, path)
        loaded = load_baseline(path)

        assert len(loaded.results) == 1
        assert loaded.results[0].faithfulness_score == 0.8


# =============================================================================
# compare
# =============================================================================


class TestComparePass:
    """compare() の pass ケース。"""

    def test_pass_when_scores_equal(self) -> None:
        """スコアが同じ場合に pass すること。"""
        current = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert result.passed is True

    def test_pass_when_scores_improved(self) -> None:
        """スコアが改善された場合に pass すること。"""
        current = _make_eval_run_result(faithfulness_mean=0.95, relevance_mean=0.9)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert result.passed is True

    def test_pass_when_drop_within_threshold(self) -> None:
        """低下が閾値以内の場合に pass すること。"""
        current = _make_eval_run_result(faithfulness_mean=0.86, relevance_mean=0.76)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert result.passed is True


class TestCompareFail:
    """compare() の fail ケース。"""

    def test_fail_when_faithfulness_drops_below_threshold(self) -> None:
        """faithfulness が閾値を超えて低下した場合に fail すること。"""
        current = _make_eval_run_result(faithfulness_mean=0.8, relevance_mean=0.8)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert result.passed is False

    def test_fail_when_relevance_drops_below_threshold(self) -> None:
        """relevance が閾値を超えて低下した場合に fail すること。"""
        current = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.7)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert result.passed is False

    def test_fail_when_both_metrics_drop(self) -> None:
        """両方のメトリクスが低下した場合に fail すること。"""
        current = _make_eval_run_result(faithfulness_mean=0.8, relevance_mean=0.7)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert result.passed is False


class TestCompareEdgeCases:
    """compare() のエッジケース。"""

    def test_custom_thresholds(self) -> None:
        """カスタム閾値が適用されること。"""
        current = _make_eval_run_result(faithfulness_mean=0.8)
        baseline = _make_eval_run_result(faithfulness_mean=0.9)

        # デフォルト閾値 (0.05) では fail だが、0.15 なら pass
        thresholds = RegressionThresholds(faithfulness_drop=0.15)
        result = compare(current, baseline, thresholds=thresholds)

        assert result.passed is True

    def test_pass_when_baseline_summary_is_none(self) -> None:
        """baseline に summary がない場合はスキップ (pass) すること。"""
        current = _make_eval_run_result(faithfulness_mean=0.9)
        baseline = _make_eval_run_result()  # summary なし

        result = compare(current, baseline)

        assert result.passed is True

    def test_pass_when_current_summary_is_none(self) -> None:
        """current に summary がない場合はスキップ (pass) すること。"""
        current = _make_eval_run_result()  # summary なし
        baseline = _make_eval_run_result(faithfulness_mean=0.9)

        result = compare(current, baseline)

        assert result.passed is True

    def test_details_contain_pass_messages(self) -> None:
        """details に pass メッセージが含まれること。"""
        current = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert len(result.details) > 0

    def test_details_contain_fail_messages(self) -> None:
        """details に fail メッセージが含まれること。"""
        current = _make_eval_run_result(faithfulness_mean=0.8)
        baseline = _make_eval_run_result(faithfulness_mean=0.9)

        result = compare(current, baseline)

        assert any("faithfulness" in d.lower() for d in result.details)

    def test_scores_are_stored_in_result(self) -> None:
        """current/baseline のスコアが結果に格納されること。"""
        current = _make_eval_run_result(faithfulness_mean=0.85, relevance_mean=0.75)
        baseline = _make_eval_run_result(faithfulness_mean=0.9, relevance_mean=0.8)

        result = compare(current, baseline)

        assert result.current_faithfulness == 0.85
        assert result.baseline_faithfulness == 0.9
        assert result.current_relevance == 0.75
        assert result.baseline_relevance == 0.8

    def test_pass_when_both_summaries_none(self) -> None:
        """両方とも summary が None の場合に pass すること。"""
        current = _make_eval_run_result()
        baseline = _make_eval_run_result()

        result = compare(current, baseline)

        assert result.passed is True

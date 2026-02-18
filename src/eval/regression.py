"""回帰テスト。

前回の評価結果 (ベースライン) と現在の結果を比較し、品質低下を検出する。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.eval import RegressionError
from src.eval.runner import EvalRunResult


class RegressionThresholds(BaseModel):
    """回帰テストの閾値。

    各メトリクスの許容低下幅を定義する。
    baseline.mean - current.mean がこの値を超えた場合に fail となる。

    Attributes:
        faithfulness_drop: 忠実性の許容低下幅。
        relevance_drop: 関連性の許容低下幅。
    """

    faithfulness_drop: float = 0.05
    relevance_drop: float = 0.05


class RegressionResult(BaseModel):
    """回帰テストの結果。

    Attributes:
        passed: テストが pass したかどうか。
        details: 各メトリクスの pass/fail 理由。
        current_faithfulness: 現在の忠実性スコア。
        baseline_faithfulness: ベースラインの忠実性スコア。
        current_relevance: 現在の関連性スコア。
        baseline_relevance: ベースラインの関連性スコア。
    """

    passed: bool
    details: list[str]
    current_faithfulness: float | None = None
    baseline_faithfulness: float | None = None
    current_relevance: float | None = None
    baseline_relevance: float | None = None


def load_baseline(path: Path) -> EvalRunResult:
    """ベースライン結果を JSON ファイルから読み込む。

    Args:
        path: JSON ファイルのパス。

    Returns:
        読み込んだ EvalRunResult。

    Raises:
        RegressionError: ファイルが存在しない、JSON が不正、またはバリデーションに失敗した場合。
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise RegressionError(f"ベースラインファイルが見つかりません: {path}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RegressionError(f"JSON のパースに失敗しました: {path}") from e

    try:
        return EvalRunResult.model_validate(data)
    except ValidationError as e:
        raise RegressionError(f"ベースラインのバリデーションに失敗しました: {e}") from e


def save_baseline(result: EvalRunResult, path: Path) -> None:
    """ベースライン結果を JSON ファイルに保存する。

    Args:
        result: 保存する評価結果。
        path: 出力先の JSON ファイルパス。
    """
    data = result.model_dump(mode="json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compare(
    current: EvalRunResult,
    baseline: EvalRunResult,
    thresholds: RegressionThresholds | None = None,
) -> RegressionResult:
    """現在の結果をベースラインと比較する。

    各メトリクスについて baseline.mean - current.mean が閾値を超えた場合に fail とする。
    baseline または current に summary がないメトリクスはスキップ (pass 扱い)。

    Args:
        current: 現在の評価結果。
        baseline: ベースラインの評価結果。
        thresholds: 回帰判定の閾値。None の場合はデフォルト値を使用。

    Returns:
        pass/fail と詳細情報を含む RegressionResult。
    """
    if thresholds is None:
        thresholds = RegressionThresholds()

    details: list[str] = []
    passed = True

    # Faithfulness
    current_faithfulness: float | None = None
    baseline_faithfulness: float | None = None

    if current.faithfulness_summary is not None and baseline.faithfulness_summary is not None:
        current_faithfulness = current.faithfulness_summary.mean
        baseline_faithfulness = baseline.faithfulness_summary.mean
        drop = baseline_faithfulness - current_faithfulness

        if drop > thresholds.faithfulness_drop:
            passed = False
            details.append(
                f"faithfulness: FAIL (baseline={baseline_faithfulness:.4f}, "
                f"current={current_faithfulness:.4f}, drop={drop:.4f}, "
                f"threshold={thresholds.faithfulness_drop:.4f})"
            )
        else:
            details.append(
                f"faithfulness: PASS (baseline={baseline_faithfulness:.4f}, "
                f"current={current_faithfulness:.4f})"
            )
    else:
        details.append("faithfulness: SKIP (summary not available)")

    # Relevance
    current_relevance: float | None = None
    baseline_relevance: float | None = None

    if current.relevance_summary is not None and baseline.relevance_summary is not None:
        current_relevance = current.relevance_summary.mean
        baseline_relevance = baseline.relevance_summary.mean
        drop = baseline_relevance - current_relevance

        if drop > thresholds.relevance_drop:
            passed = False
            details.append(
                f"relevance: FAIL (baseline={baseline_relevance:.4f}, "
                f"current={current_relevance:.4f}, drop={drop:.4f}, "
                f"threshold={thresholds.relevance_drop:.4f})"
            )
        else:
            details.append(
                f"relevance: PASS (baseline={baseline_relevance:.4f}, "
                f"current={current_relevance:.4f})"
            )
    else:
        details.append("relevance: SKIP (summary not available)")

    return RegressionResult(
        passed=passed,
        details=details,
        current_faithfulness=current_faithfulness,
        baseline_faithfulness=baseline_faithfulness,
        current_relevance=current_relevance,
        baseline_relevance=baseline_relevance,
    )

"""評価メトリクス共通ユーティリティ。"""

from __future__ import annotations

import re

from src.eval import MetricError

_SCORE_PATTERN = re.compile(r"Score:\s*(-?\d+(?:\.\d+)?)")
_REASON_PATTERN = re.compile(r"Reason:\s*(.*)", re.DOTALL)


def parse_evaluation_response(content: str) -> tuple[float, str]:
    """LLM-as-judge の評価レスポンスをパースする。

    期待フォーマット::

        Score: <float>
        Reason: <text>

    Args:
        content: LLM レスポンスの生テキスト。

    Returns:
        (score, reason) のタプル。score は 0.0-1.0 にクランプされる。
        Reason が存在しない場合は空文字列。

    Raises:
        MetricError: Score のパースに失敗した場合。
    """
    score_match = _SCORE_PATTERN.search(content)
    if not score_match:
        raise MetricError(f"Score を抽出できません: {content!r}")

    try:
        score = float(score_match.group(1))
    except ValueError as e:
        raise MetricError(f"Score が数値ではありません: {score_match.group(1)!r}") from e

    score = max(0.0, min(1.0, score))

    reason_match = _REASON_PATTERN.search(content)
    reason = reason_match.group(1).strip() if reason_match else ""

    return score, reason

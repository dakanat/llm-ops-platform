"""評価実行エンジン。

データセットに対してメトリクスを一括実行し、結果をサマリとともに返す。
"""

from __future__ import annotations

from pydantic import BaseModel

from src.eval.datasets import EvalDataset, EvalExample
from src.eval.metrics.faithfulness import FaithfulnessMetric
from src.eval.metrics.relevance import RelevanceMetric


class ExampleResult(BaseModel):
    """1件の評価結果。

    Attributes:
        example: 評価対象のサンプル。
        faithfulness_score: 忠実性スコア (未評価時は None)。
        relevance_score: 関連性スコア (未評価時は None)。
        latency_seconds: レイテンシ秒数 (未計測時は None)。
        error: エラーメッセージ (正常時は None)。
    """

    example: EvalExample
    faithfulness_score: float | None = None
    relevance_score: float | None = None
    latency_seconds: float | None = None
    error: str | None = None


class MetricSummary(BaseModel):
    """メトリクスの集計結果。

    Attributes:
        mean: 平均値。
        count: サンプル数。
    """

    mean: float
    count: int


class EvalRunResult(BaseModel):
    """評価実行全体の結果。

    Attributes:
        dataset_name: データセット名。
        results: 各サンプルの評価結果。
        faithfulness_summary: 忠実性の集計 (未評価時は None)。
        relevance_summary: 関連性の集計 (未評価時は None)。
        latency_summary: レイテンシの集計 (未計測時は None)。
    """

    dataset_name: str
    results: list[ExampleResult]
    faithfulness_summary: MetricSummary | None = None
    relevance_summary: MetricSummary | None = None
    latency_summary: MetricSummary | None = None


class EvalRunner:
    """評価実行エンジン。

    メトリクスをコンストラクタ注入し、データセットに対して一括評価を実行する。
    1件のエラーで全体が止まらないよう、エラーは ExampleResult.error に記録して続行する。
    """

    def __init__(
        self,
        faithfulness_metric: FaithfulnessMetric | None = None,
        relevance_metric: RelevanceMetric | None = None,
    ) -> None:
        self._faithfulness_metric = faithfulness_metric
        self._relevance_metric = relevance_metric

    async def run(self, dataset: EvalDataset) -> EvalRunResult:
        """データセットに対して評価を実行する。

        Args:
            dataset: 評価対象のデータセット。

        Returns:
            全サンプルの評価結果とサマリを含む EvalRunResult。
        """
        results: list[ExampleResult] = []
        for example in dataset.examples:
            result = await self._evaluate_example(example)
            results.append(result)

        faithfulness_scores = [
            r.faithfulness_score for r in results if r.faithfulness_score is not None
        ]
        relevance_scores = [r.relevance_score for r in results if r.relevance_score is not None]

        return EvalRunResult(
            dataset_name=dataset.name,
            results=results,
            faithfulness_summary=self._compute_summary(faithfulness_scores),
            relevance_summary=self._compute_summary(relevance_scores),
        )

    async def _evaluate_example(self, example: EvalExample) -> ExampleResult:
        """1件のサンプルを評価する。

        Args:
            example: 評価対象のサンプル。

        Returns:
            評価結果。エラー時は error フィールドにメッセージを記録。
        """
        faithfulness_score: float | None = None
        relevance_score: float | None = None

        try:
            if self._faithfulness_metric is not None:
                faith_result = await self._faithfulness_metric.evaluate(
                    context=example.context, answer=example.answer
                )
                faithfulness_score = faith_result.score

            if self._relevance_metric is not None:
                rel_result = await self._relevance_metric.evaluate(
                    query=example.query, answer=example.answer
                )
                relevance_score = rel_result.score
        except Exception as e:
            return ExampleResult(example=example, error=str(e))

        return ExampleResult(
            example=example,
            faithfulness_score=faithfulness_score,
            relevance_score=relevance_score,
        )

    def _compute_summary(self, scores: list[float]) -> MetricSummary | None:
        """スコアリストからサマリを計算する。

        Args:
            scores: スコアのリスト。

        Returns:
            平均値とカウントを含む MetricSummary。空リストの場合は None。
        """
        if not scores:
            return None
        return MetricSummary(mean=sum(scores) / len(scores), count=len(scores))

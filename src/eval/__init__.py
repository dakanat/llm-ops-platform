"""評価モジュール。

RAGパイプラインの出力品質を定量評価するためのメトリクスと実行エンジンを提供する。
"""


class EvalError(Exception):
    """評価処理に関する基底エラー。"""


class MetricError(EvalError):
    """個別メトリクスの評価に関するエラー。"""

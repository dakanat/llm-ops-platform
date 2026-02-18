"""評価モジュール。

RAGパイプラインの出力品質を定量評価するためのメトリクスと実行エンジンを提供する。
"""


class EvalError(Exception):
    """評価処理に関する基底エラー。"""


class MetricError(EvalError):
    """個別メトリクスの評価に関するエラー。"""


class DatasetError(EvalError):
    """データセットの読み込み・バリデーションに関するエラー。"""


class RegressionError(EvalError):
    """回帰テストに関するエラー。"""


class SyntheticDataError(EvalError):
    """合成データ生成に関するエラー。"""

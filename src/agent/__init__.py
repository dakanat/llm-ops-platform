"""Agent モジュール。

Agent Runtime、ツール基盤、ガードレールを提供する。
"""


class ToolError(Exception):
    """ツール処理に関する基底エラー。"""


class ToolNotFoundError(ToolError):
    """ツールがレジストリに未登録の場合のエラー。"""


class DuplicateToolError(ToolError):
    """同名ツールが既に登録済みの場合のエラー。"""


class ToolExecutionError(ToolError):
    """ツール実行時のエラー。"""

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


class AgentError(Exception):
    """Agent 実行に関する基底エラー。"""


class AgentParseError(AgentError):
    """LLM 出力のパースに失敗した場合のエラー。"""


class GuardrailError(AgentError):
    """ガードレールによって入力または出力がブロックされた場合のエラー。"""

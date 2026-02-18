"""LLM モジュール。

LLM プロバイダ抽象化、ルーティング、プロンプト管理を提供する。
"""


class PromptError(Exception):
    """プロンプト処理に関する基底エラー。"""


class PromptNotFoundError(PromptError):
    """テンプレートが未登録の場合のエラー。"""


class PromptRenderError(PromptError):
    """テンプレートの変数埋め込みに失敗した場合のエラー。"""


class CacheError(Exception):
    """セマンティックキャッシュに関するエラー。"""

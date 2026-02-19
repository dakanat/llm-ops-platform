"""LLM token counting and pre-request usage estimation.

tiktoken を使ってプロンプトのトークン数をカウントし、
API呼び出し前のコスト推定やトークンバジェット管理を可能にする。
"""

from __future__ import annotations

import tiktoken

from src.llm.providers.base import ChatMessage, TokenUsage

DEFAULT_ENCODING = "cl100k_base"

_MODEL_ENCODING_MAP: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
}

_TOKENS_PER_MESSAGE = 4
"""Per-message overhead: ``<|im_start|>{role}\\n{content}<|im_end|>\\n``."""

_TOKENS_PER_REPLY = 3
"""Assistant reply priming tokens."""


class TokenCounter:
    """tiktoken を用いたトークンカウンター。

    テキストやメッセージリストのトークン数をカウントし、
    ``TokenUsage`` として返すことで ``CostTracker`` と組み合わせた
    事前コスト推定を実現する。
    """

    def __init__(self) -> None:
        self._encoding_cache: dict[str, tiktoken.Encoding] = {}

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        """プロバイダプレフィックスとサフィックスタグを除去する。

        Args:
            model: モデル名（例: ``"openai/gpt-4:free"``）。

        Returns:
            正規化されたモデル名（例: ``"gpt-4"``）。
        """
        # Strip provider prefix (e.g. "openai/", "anthropic/")
        if "/" in model:
            model = model.split("/", 1)[1]
        # Strip suffix tag (e.g. ":free", ":beta")
        if ":" in model:
            model = model.split(":")[0]
        return model

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """モデルに対応するエンコーディングを取得する（キャッシュ付き）。

        解決順序:
            1. ``tiktoken.encoding_for_model`` で既知モデルを検索
            2. ``_MODEL_ENCODING_MAP`` でプレフィックスマッチ
            3. ``DEFAULT_ENCODING`` にフォールバック

        Args:
            model: モデル名（OpenRouter 形式も可）。

        Returns:
            対応する tiktoken エンコーディング。
        """
        normalized = self._normalize_model_name(model)

        if normalized in self._encoding_cache:
            return self._encoding_cache[normalized]

        # 1. Try tiktoken's built-in model registry
        try:
            encoding = tiktoken.encoding_for_model(normalized)
            self._encoding_cache[normalized] = encoding
            return encoding
        except KeyError:
            pass

        # 2. Match against _MODEL_ENCODING_MAP by prefix
        for prefix, encoding_name in _MODEL_ENCODING_MAP.items():
            if normalized.startswith(prefix):
                encoding = tiktoken.get_encoding(encoding_name)
                self._encoding_cache[normalized] = encoding
                return encoding

        # 3. Fallback to default encoding
        encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
        self._encoding_cache[normalized] = encoding
        return encoding

    def count_text(self, text: str, model: str) -> int:
        """テキストのトークン数をカウントする。

        Args:
            text: カウント対象のテキスト。
            model: トークナイザ選択に使うモデル名。

        Returns:
            トークン数。
        """
        encoding = self._get_encoding(model)
        return len(encoding.encode(text))

    def count_messages(self, messages: list[ChatMessage], model: str) -> int:
        """メッセージリストのトークン数をカウントする。

        OpenAI のメッセージフォーマットオーバーヘッドを含む。
        各メッセージに ``_TOKENS_PER_MESSAGE`` トークン、
        全体に ``_TOKENS_PER_REPLY`` トークンを加算する。

        Args:
            messages: チャットメッセージのリスト。
            model: トークナイザ選択に使うモデル名。

        Returns:
            メッセージオーバーヘッドを含むトークン数。
        """
        encoding = self._get_encoding(model)
        num_tokens = 0
        for message in messages:
            num_tokens += _TOKENS_PER_MESSAGE
            num_tokens += len(encoding.encode(message.role.value))
            num_tokens += len(encoding.encode(message.content))
        num_tokens += _TOKENS_PER_REPLY
        return num_tokens

    def estimate_prompt_usage(self, messages: list[ChatMessage], model: str) -> TokenUsage:
        """メッセージリストから ``TokenUsage`` を推定する。

        ``completion_tokens`` は 0 となる（事前推定のため）。
        ``CostTracker.calculate_cost`` と組み合わせて
        API呼び出し前のコスト推定に使用できる。

        Args:
            messages: チャットメッセージのリスト。
            model: トークナイザ選択に使うモデル名。

        Returns:
            推定プロンプトトークン数を含む ``TokenUsage``。
        """
        prompt_tokens = self.count_messages(messages, model)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        )

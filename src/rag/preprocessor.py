"""Text preprocessor for RAG pipeline document normalization."""

from __future__ import annotations

import re
import unicodedata


class PreprocessingError(Exception):
    """前処理に関するエラー。"""


class Preprocessor:
    """テキスト前処理を行うクラス。

    Unicode NFKC 正規化、空白正規化、改行正規化を適用する。
    """

    def normalize(self, text: str) -> str:
        """テキストを正規化する。

        処理内容:
        1. Unicode NFKC 正規化 (全角→半角)
        2. 連続スペース / タブ → 単一スペース (改行は保持)
        3. 3 つ以上の連続改行 → 2 つの改行
        4. 先頭・末尾の空白を除去

        Args:
            text: 正規化するテキスト。

        Returns:
            正規化済みテキスト。

        Raises:
            PreprocessingError: 空文字または空白のみの場合。
        """
        if not text or not text.strip():
            raise PreprocessingError("text must not be empty")

        # NFKC 正規化
        result = unicodedata.normalize("NFKC", text)

        # 連続スペース / タブ → 単一スペース (改行は保持)
        result = re.sub(r"[^\S\n]+", " ", result)

        # 3 つ以上の連続改行 → 2 つ
        result = re.sub(r"\n{3,}", "\n\n", result)

        # 先頭・末尾の空白を除去
        result = result.strip()

        return result

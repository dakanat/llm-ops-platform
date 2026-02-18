"""Tests for Preprocessor.normalize()."""

from __future__ import annotations

import pytest
from src.rag.preprocessor import PreprocessingError, Preprocessor


class TestPreprocessorNormalize:
    """Preprocessor.normalize() のテスト。"""

    def test_nfkc_normalizes_fullwidth_to_halfwidth(self) -> None:
        """全角英数字が半角に正規化されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("Ｈｅｌｌｏ　Ｗｏｒｌｄ　１２３")

        assert result == "Hello World 123"

    def test_collapses_multiple_spaces_to_single(self) -> None:
        """連続スペースが単一スペースに正規化されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello   world")

        assert result == "hello world"

    def test_preserves_single_newlines(self) -> None:
        """単一改行が保持されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello\nworld")

        assert result == "hello\nworld"

    def test_preserves_double_newlines(self) -> None:
        """二重改行 (段落区切り) が保持されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello\n\nworld")

        assert result == "hello\n\nworld"

    def test_collapses_triple_newlines_to_double(self) -> None:
        """3 つ以上の改行が 2 つに正規化されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("hello\n\n\n\nworld")

        assert result == "hello\n\nworld"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        """先頭・末尾の空白が除去されること。"""
        preprocessor = Preprocessor()

        result = preprocessor.normalize("  hello world  ")

        assert result == "hello world"

    def test_empty_string_raises_error(self) -> None:
        """空文字で PreprocessingError が発生すること。"""
        preprocessor = Preprocessor()

        with pytest.raises(PreprocessingError, match="empty"):
            preprocessor.normalize("")

    def test_whitespace_only_raises_error(self) -> None:
        """空白のみの文字列で PreprocessingError が発生すること。"""
        preprocessor = Preprocessor()

        with pytest.raises(PreprocessingError, match="empty"):
            preprocessor.normalize("   \n\n  ")

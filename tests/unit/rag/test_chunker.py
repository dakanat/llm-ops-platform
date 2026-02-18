"""Tests for recursive character text splitter."""

from __future__ import annotations

import pytest
from src.rag.chunker import ChunkingError, RecursiveCharacterSplitter, TextChunk


class TestTextChunk:
    """TextChunk データモデルのテスト。"""

    def test_creates_with_all_fields(self) -> None:
        """全フィールドで生成できること。"""
        chunk = TextChunk(content="hello", index=0, start=0, end=5)

        assert chunk.content == "hello"
        assert chunk.index == 0
        assert chunk.start == 0
        assert chunk.end == 5


class TestRecursiveCharacterSplitterInit:
    """RecursiveCharacterSplitter 初期化のテスト。"""

    def test_creates_with_default_parameters(self) -> None:
        """デフォルト値 (512, 64) で生成できること。"""
        splitter = RecursiveCharacterSplitter()

        assert splitter._chunk_size == 512
        assert splitter._chunk_overlap == 64

    def test_creates_with_custom_parameters(self) -> None:
        """カスタムパラメータで生成できること。"""
        splitter = RecursiveCharacterSplitter(chunk_size=256, chunk_overlap=32)

        assert splitter._chunk_size == 256
        assert splitter._chunk_overlap == 32

    def test_creates_with_custom_separators(self) -> None:
        """カスタムセパレータで生成できること。"""
        seps = ["\n", " ", ""]
        splitter = RecursiveCharacterSplitter(separators=seps)

        assert splitter._separators == seps

    def test_raises_on_zero_chunk_size(self) -> None:
        """chunk_size=0 で ChunkingError が発生すること。"""
        with pytest.raises(ChunkingError):
            RecursiveCharacterSplitter(chunk_size=0)

    def test_raises_on_negative_chunk_size(self) -> None:
        """負の chunk_size で ChunkingError が発生すること。"""
        with pytest.raises(ChunkingError):
            RecursiveCharacterSplitter(chunk_size=-1)

    def test_raises_on_negative_overlap(self) -> None:
        """負の overlap で ChunkingError が発生すること。"""
        with pytest.raises(ChunkingError):
            RecursiveCharacterSplitter(chunk_overlap=-1)

    def test_raises_on_overlap_ge_chunk_size(self) -> None:
        """overlap >= chunk_size で ChunkingError が発生すること。"""
        with pytest.raises(ChunkingError):
            RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ChunkingError):
            RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=200)


class TestRecursiveCharacterSplitterSplit:
    """RecursiveCharacterSplitter.split() 分割動作のテスト。"""

    def test_split_returns_list_of_text_chunks(self) -> None:
        """戻り値が list[TextChunk] であること。"""
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=0)
        result = splitter.split("a" * 100)

        assert isinstance(result, list)
        assert all(isinstance(c, TextChunk) for c in result)

    def test_split_short_text_returns_single_chunk(self) -> None:
        """chunk_size 以下のテキストは 1 チャンクになること。"""
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0)
        result = splitter.split("short text")

        assert len(result) == 1
        assert result[0].content == "short text"

    def test_split_at_paragraph_boundary(self) -> None:
        """\\n\\n で分割されること。"""
        text = "First paragraph." + "\n\n" + "Second paragraph."
        splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) == 2
        assert result[0].content == "First paragraph."
        assert result[1].content == "Second paragraph."

    def test_split_at_line_boundary(self) -> None:
        """\\n で分割されること。"""
        text = "Line one." + "\n" + "Line two."
        splitter = RecursiveCharacterSplitter(chunk_size=15, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) == 2
        assert result[0].content == "Line one."
        assert result[1].content == "Line two."

    def test_split_at_sentence_boundary(self) -> None:
        """ ". " で分割されること (英語)。"""
        text = "First sentence. Second sentence."
        splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) == 2
        assert "First sentence." in result[0].content
        assert "Second sentence." in result[1].content

    def test_split_at_japanese_sentence_boundary(self) -> None:
        """「。」で分割されること (日本語)。"""
        text = "最初の文章です。次の文章です。"
        splitter = RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) == 2
        assert "最初の文章です。" in result[0].content
        assert "次の文章です。" in result[1].content

    def test_split_at_space_boundary(self) -> None:
        """スペースで分割されること。"""
        text = "word " * 20  # 100 chars
        splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.content) <= 30

    def test_split_character_level_as_last_resort(self) -> None:
        """セパレータなしの場合、文字レベルで分割されること。"""
        text = "a" * 100  # no separators present
        splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.content) <= 30

    def test_each_chunk_within_size_limit(self) -> None:
        """全チャンクが chunk_size 以下であること。"""
        text = "Hello world. " * 100
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
        result = splitter.split(text)

        for chunk in result:
            assert len(chunk.content) <= 50

    def test_chunk_indices_are_sequential(self) -> None:
        """index が 0 から連番であること。"""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
        result = splitter.split(text)

        for i, chunk in enumerate(result):
            assert chunk.index == i


class TestRecursiveCharacterSplitterOverlap:
    """オーバーラップ動作のテスト。"""

    def test_overlap_content_appears_in_adjacent_chunks(self) -> None:
        """隣接チャンク間に重複テキストが存在すること。"""
        text = "abcdefghij" * 5  # 50 chars
        splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=5)
        result = splitter.split(text)

        assert len(result) >= 2
        for i in range(len(result) - 1):
            current_end = result[i].content[-5:]
            next_start = result[i + 1].content[:5]
            assert current_end == next_start

    def test_zero_overlap_produces_no_duplication(self) -> None:
        """overlap=0 で重複がないこと。"""
        text = "a" * 100
        splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=0)
        result = splitter.split(text)

        total_content = "".join(c.content for c in result)
        assert total_content == text


class TestRecursiveCharacterSplitterEdgeCases:
    """エッジケースのテスト。"""

    def test_empty_string_returns_empty_list(self) -> None:
        """空文字 → [] を返すこと。"""
        splitter = RecursiveCharacterSplitter()
        result = splitter.split("")

        assert result == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        """空白のみ → [] を返すこと。"""
        splitter = RecursiveCharacterSplitter()
        result = splitter.split("   \n\n  \t  ")

        assert result == []

    def test_exact_chunk_size_returns_single_chunk(self) -> None:
        """ちょうど chunk_size の長さ → 1 チャンクになること。"""
        splitter = RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=0)
        result = splitter.split("a" * 10)

        assert len(result) == 1
        assert result[0].content == "a" * 10

    def test_very_long_text_produces_many_chunks(self) -> None:
        """長大テキスト → 多数のチャンクが生成されること。"""
        text = "word " * 10000  # 50000 chars
        splitter = RecursiveCharacterSplitter(chunk_size=512, chunk_overlap=64)
        result = splitter.split(text)

        assert len(result) > 50
        for chunk in result:
            assert len(chunk.content) <= 512


class TestRecursiveCharacterSplitterJapanese:
    """日本語テキストのテスト。"""

    def test_japanese_text_splits_at_sentence_boundary(self) -> None:
        """日本語テキストが「。」で分割されること。"""
        text = "吾輩は猫である。名前はまだない。どこで生まれたか見当がつかぬ。"
        splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) >= 2
        assert result[0].content.endswith("。")

    def test_japanese_paragraph_split(self) -> None:
        """日本語テキストが \\n\\n で分割されること。"""
        text = "最初の段落です。\n\n次の段落です。"
        splitter = RecursiveCharacterSplitter(chunk_size=15, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) == 2
        assert result[0].content == "最初の段落です。"
        assert result[1].content == "次の段落です。"

    def test_mixed_japanese_english_text(self) -> None:
        """日英混在テキストが分割されること。"""
        text = "Hello World.\n\nこんにちは世界。"
        splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) == 2
        assert "Hello" in result[0].content
        assert "こんにちは" in result[1].content

    def test_japanese_multibyte_characters_counted_correctly(self) -> None:
        """日本語文字が文字数 (バイト数ではなく) で正しくカウントされること。"""
        # 「あ」は3バイトだが1文字
        text = "あ" * 10
        splitter = RecursiveCharacterSplitter(chunk_size=5, chunk_overlap=0)
        result = splitter.split(text)

        assert len(result) == 2
        assert result[0].content == "あ" * 5
        assert result[1].content == "あ" * 5

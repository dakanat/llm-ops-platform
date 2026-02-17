"""Recursive character text splitter for RAG pipeline chunking."""

from __future__ import annotations

from pydantic import BaseModel


class ChunkingError(Exception):
    """チャンキング処理に関するエラー。"""


class TextChunk(BaseModel):
    """分割されたテキストチャンク。

    Attributes:
        content: チャンクのテキスト。
        index: 0 始まりの通し番号。
        start: 元テキスト内の開始文字位置。
        end: 元テキスト内の終了文字位置 (排他的)。
    """

    content: str
    index: int
    start: int
    end: int


class RecursiveCharacterSplitter:
    """再帰的文字分割器。

    セパレータリストを優先度順に試し、テキストを ``chunk_size`` 以下の
    チャンクに分割する。隣接チャンク間で ``chunk_overlap`` 文字の
    重複を持たせる。

    Attributes:
        DEFAULT_SEPARATORS: デフォルトのセパレータリスト (粒度の粗い順)。
    """

    DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", "。", ". ", "、", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    def split(self, text: str) -> list[TextChunk]:
        """テキストをチャンクに分割する。

        Args:
            text: 分割するテキスト。空文字・空白のみの場合は空リストを返す。

        Returns:
            TextChunk のリスト。各チャンクに index, start, end を付与。
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split_recursive(text, self._separators)

        # 空白のみのチャンクを除去
        raw_chunks = [c for c in raw_chunks if c.strip()]

        # TextChunk オブジェクトに変換 (start/end を計算)
        result: list[TextChunk] = []
        search_start = 0
        for i, content in enumerate(raw_chunks):
            start = text.find(content, search_start)
            if start == -1:
                # overlap により元テキストの部分文字列ではない場合がある
                # その場合は直前の検索位置から推定
                start = search_start
            end = start + len(content)
            result.append(TextChunk(content=content, index=i, start=start, end=end))
            # 次の検索はオーバーラップを考慮して現チャンクの途中から
            search_start = start + len(content) - self._chunk_overlap
            if search_start < 0:
                search_start = 0

        return result

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """セパレータリストを順に試し、再帰的にテキストを分割する。"""
        if len(text) <= self._chunk_size:
            return [text]

        # 使用可能なセパレータを探す
        separator = ""
        remaining_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        # セパレータで分割
        if separator == "":
            splits = list(text)
        else:
            raw_splits = text.split(separator)
            keep_separator = separator.strip() != ""
            if keep_separator:
                # 句読点系セパレータ ("。", ". " 等) は直前の分割片に付与して保持
                splits = []
                for j, part in enumerate(raw_splits):
                    if j < len(raw_splits) - 1:
                        splits.append(part + separator)
                    else:
                        splits.append(part)
            else:
                splits = raw_splits
            # 空要素を除去
            splits = [s for s in splits if s]

        # マージしてチャンクを構築
        join_sep = "" if separator.strip() != "" else separator
        merged = self._merge_splits(splits, join_sep)

        # chunk_size を超えるチャンクがあれば、より細かいセパレータで再帰
        final: list[str] = []
        for chunk in merged:
            if len(chunk) <= self._chunk_size:
                final.append(chunk)
            elif remaining_separators:
                final.extend(self._split_recursive(chunk, remaining_separators))
            else:
                # 最終手段: 文字レベルで分割
                final.extend(self._split_recursive(chunk, [""]))
        return final

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """小さな分割片を chunk_size 以下になるよう結合し、overlap を適用する。"""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            sep_len = len(separator) if current else 0
            candidate_len = current_len + split_len + sep_len

            if candidate_len > self._chunk_size and current:
                # 現在のチャンクを確定
                chunk_text = separator.join(current)
                chunks.append(chunk_text)

                # overlap: 末尾から chunk_overlap 文字分をキープ
                if self._chunk_overlap > 0:
                    current, current_len = self._build_overlap(current, separator, chunk_text)
                else:
                    current = []
                    current_len = 0

            current.append(split)
            if current_len > 0:
                current_len += len(separator) + split_len
            else:
                current_len = split_len

        # 残りを追加
        if current:
            chunks.append(separator.join(current))

        return chunks

    def _build_overlap(
        self, parts: list[str], separator: str, chunk_text: str
    ) -> tuple[list[str], int]:
        """前チャンクの末尾から overlap 分の parts を保持する。"""
        overlap_parts: list[str] = []
        overlap_len = 0
        for part in reversed(parts):
            added_len = len(part) + (len(separator) if overlap_parts else 0)
            if overlap_len + added_len > self._chunk_overlap:
                break
            overlap_parts.insert(0, part)
            overlap_len += added_len

        return overlap_parts, overlap_len

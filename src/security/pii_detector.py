"""PII検出・マスキング。

メールアドレス、電話番号、マイナンバー (12桁)、クレジットカード番号 (16桁)、
住所 (都道府県+市区町村) を正規表現で検出し、``[TYPE]`` ラベルでマスキングする。

日本語の氏名検出は誤検出率が高いため対象外とする (将来NERで対応)。
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel


class PIIType(StrEnum):
    """PII種別。

    email: メールアドレス。
    phone: 電話番号。
    my_number: マイナンバー (12桁)。
    credit_card: クレジットカード番号 (16桁)。
    address: 住所 (都道府県+市区町村)。
    """

    EMAIL = "email"
    PHONE = "phone"
    MY_NUMBER = "my_number"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"


class PIIMatch(BaseModel):
    """PII検出結果の1件。

    マッチしたテキスト自体は保持しない (ログへのPII漏洩を防止)。

    Attributes:
        pii_type: PII種別。
        start: マッチ開始位置。
        end: マッチ終了位置。
    """

    pii_type: PIIType
    start: int
    end: int


class PIIDetectionResult(BaseModel):
    """PII検出の全体結果。

    Attributes:
        has_pii: PIIが検出されたかどうか。
        matches: 検出されたPIIのリスト。
        masked_text: PIIをマスキング済みのテキスト。
    """

    has_pii: bool
    matches: list[PIIMatch]
    masked_text: str


# --- Precompiled regex patterns ---

# クレジットカード (16桁): マイナンバー (12桁) より先にチェックし重複を防ぐ
_CREDIT_CARD_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")

# マイナンバー (12桁): 前後が数字でないことを確認
_MY_NUMBER_RE = re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)")

# メールアドレス
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# 電話番号: 0始まり (国内) または +81 (国際) で始まる
# \b の代わりに (?!\d) を使用 (CJK文字の直前で \b が機能しないため)
_PHONE_RE = re.compile(
    r"(?:\+81[-\s]?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4})"  # +81 国際形式
    r"|"
    r"(?:0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4})(?!\d)",  # 0始まり国内形式
)

# 住所: 都道府県 + 市区町村 (市|区|町|村|郡)
_ADDRESS_RE = re.compile(
    r"(?:北海道|東京都|(?:大阪|京都)府|.{2,3}県)"
    r"(?:[\u4e00-\u9fff]{1,5}(?:市|区|町|村|郡))",
)

# パターンとPII種別の対応 (longest-match-first のため CREDIT_CARD を MY_NUMBER より先に配置)
_PATTERNS: list[tuple[PIIType, re.Pattern[str]]] = [
    (PIIType.CREDIT_CARD, _CREDIT_CARD_RE),
    (PIIType.MY_NUMBER, _MY_NUMBER_RE),
    (PIIType.EMAIL, _EMAIL_RE),
    (PIIType.PHONE, _PHONE_RE),
    (PIIType.ADDRESS, _ADDRESS_RE),
]


class PIIDetector:
    """PII検出・マスキングエンジン。

    正規表現ベースで各PII種別を検出し、``[TYPE]`` ラベルでマスキングする。
    重複マッチは開始位置と長さの優先順位で解決する。

    Args:
        enabled_types: 検出対象のPII種別。None の場合はすべて有効。
    """

    def __init__(self, enabled_types: set[PIIType] | None = None) -> None:
        if enabled_types is None:
            self._patterns = list(_PATTERNS)
        else:
            self._patterns = [(t, p) for t, p in _PATTERNS if t in enabled_types]

    def detect(self, text: str) -> PIIDetectionResult:
        """テキスト中のPIIを検出しマスキングする。

        Args:
            text: 検査対象テキスト。

        Returns:
            検出結果 (マッチ一覧とマスキング済みテキスト)。
        """
        if not text:
            return PIIDetectionResult(has_pii=False, matches=[], masked_text=text)

        # 全パターンの全マッチを収集
        raw_matches: list[tuple[PIIType, int, int]] = []
        for pii_type, pattern in self._patterns:
            for m in pattern.finditer(text):
                raw_matches.append((pii_type, m.start(), m.end()))

        # 開始位置昇順、長さ降順でソートし重複を除去
        raw_matches.sort(key=lambda x: (x[1], -(x[2] - x[1])))
        resolved: list[tuple[PIIType, int, int]] = []
        prev_end = 0
        for pii_type, start, end in raw_matches:
            if start >= prev_end:
                resolved.append((pii_type, start, end))
                prev_end = end

        # マスキングテキスト生成
        matches: list[PIIMatch] = []
        parts: list[str] = []
        cursor = 0
        for pii_type, start, end in resolved:
            parts.append(text[cursor:start])
            label = f"[{pii_type.value.upper()}]"
            parts.append(label)
            matches.append(PIIMatch(pii_type=pii_type, start=start, end=end))
            cursor = end
        parts.append(text[cursor:])

        masked_text = "".join(parts)
        return PIIDetectionResult(
            has_pii=len(matches) > 0,
            matches=matches,
            masked_text=masked_text,
        )

    def mask(self, text: str) -> str:
        """テキスト中のPIIをマスキングした文字列を返す。

        ``detect(text).masked_text`` のショートカット。

        Args:
            text: 検査対象テキスト。

        Returns:
            マスキング済みテキスト。
        """
        return self.detect(text).masked_text

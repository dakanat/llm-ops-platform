"""プロンプトインジェクション検出。

命令上書き、システムプロンプト漏洩、ロール操作、デリミタインジェクションを
正規表現ベースで検出する。各パターンは動詞+対象名詞の組み合わせを必須とし、
単一キーワードによる誤検出を防ぐ。
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel


class InjectionType(StrEnum):
    """インジェクション種別。

    instruction_override: 命令上書き。
    system_prompt_leak: システムプロンプト漏洩。
    role_manipulation: ロール操作。
    delimiter_injection: デリミタインジェクション。
    """

    INSTRUCTION_OVERRIDE = "instruction_override"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    ROLE_MANIPULATION = "role_manipulation"
    DELIMITER_INJECTION = "delimiter_injection"


class RiskLevel(StrEnum):
    """リスクレベル。

    none: リスクなし。
    low: 低リスク。
    high: 高リスク。
    """

    NONE = "none"
    LOW = "low"
    HIGH = "high"


class InjectionMatch(BaseModel):
    """インジェクション検出結果の1件。

    攻撃ペイロードは監査用に保持する (PII と異なり秘匿不要)。

    Attributes:
        injection_type: インジェクション種別。
        matched_text: マッチしたテキスト。
        description: パターンの説明。
    """

    injection_type: InjectionType
    matched_text: str
    description: str


class InjectionDetectionResult(BaseModel):
    """インジェクション検出の全体結果。

    Attributes:
        is_injection: インジェクションが検出されたかどうか。
        matches: 検出されたインジェクションのリスト。
        risk_level: リスクレベル。
        reason: 最初のマッチの説明 (なければ None)。
    """

    is_injection: bool
    matches: list[InjectionMatch]
    risk_level: RiskLevel
    reason: str | None = None


# --- Precompiled regex patterns ---
# 各タプル: (InjectionType, compiled_pattern, description)

_INSTRUCTION_OVERRIDE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+"
            r"(?:instructions|rules|prompts)",
            re.IGNORECASE,
        ),
        "Instruction override: ignore previous instructions",
    ),
    (
        re.compile(
            r"forget\s+(?:your|all)\s+(?:instructions|rules)",
            re.IGNORECASE,
        ),
        "Instruction override: forget instructions",
    ),
    (
        re.compile(
            r"disregard\s+(?:all\s+)?(?:above|previous)\s+instructions",
            re.IGNORECASE,
        ),
        "Instruction override: disregard instructions",
    ),
    (
        re.compile(
            r"override\s+your\s+(?:instructions|programming)",
            re.IGNORECASE,
        ),
        "Instruction override: override programming",
    ),
    (
        re.compile(
            r"do\s+not\s+follow\s+(?:your|the)\s+instructions",
            re.IGNORECASE,
        ),
        "Instruction override: do not follow instructions",
    ),
    (
        re.compile(
            r"(?:前の|以前の|上記の)?指示を(?:無視|忘れ|無効に)",
        ),
        "命令上書き: 指示を無視/忘却",
    ),
]

_SYSTEM_PROMPT_LEAK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:show|reveal|display)\s+(?:me\s+)?your\s+"
            r"(?:system\s+)?(?:prompt|instructions)",
            re.IGNORECASE,
        ),
        "System prompt leak: show/reveal prompt",
    ),
    (
        re.compile(
            r"what\s+are\s+your\s+(?:system\s+)?(?:instructions|rules|system\s+prompt)",
            re.IGNORECASE,
        ),
        "System prompt leak: what are your instructions",
    ),
    (
        re.compile(
            r"repeat\s+your\s+(?:initial|system)\s+(?:prompt|instructions)",
            re.IGNORECASE,
        ),
        "System prompt leak: repeat prompt",
    ),
    (
        re.compile(
            r"(?:output|print)\s+everything\s+(?:above|before)",
            re.IGNORECASE,
        ),
        "System prompt leak: output everything above",
    ),
    (
        re.compile(
            r"(?:システムプロンプト|初期指示|設定内容)を(?:表示|教え|出力|見せ|開示)",
        ),
        "システムプロンプト漏洩: プロンプト表示要求",
    ),
]

_ROLE_MANIPULATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"you\s+are\s+now\s+(?:DAN|unrestricted|jailbroken|evil)",
            re.IGNORECASE,
        ),
        "Role manipulation: identity override",
    ),
    (
        re.compile(
            r"act\s+as\s+an?\s+(?:unrestricted|unfiltered)\s+(?:AI|assistant)",
            re.IGNORECASE,
        ),
        "Role manipulation: act as unrestricted",
    ),
    (
        re.compile(
            r"pretend\s+you\s*(?:\s+have)?\s+no\s+(?:restrictions|rules|filters)",
            re.IGNORECASE,
        ),
        "Role manipulation: pretend no restrictions",
    ),
    (
        re.compile(
            r"(?:enter|enable)\s+(?:DAN|developer|jailbreak)\s+mode",
            re.IGNORECASE,
        ),
        "Role manipulation: enable jailbreak mode",
    ),
    (
        re.compile(
            r"(?:制限|フィルター|安全装置)を?(?:解除|無効|外し|なし|無視)",
        ),
        "ロール操作: 制限解除要求",
    ),
]

_DELIMITER_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"</?(?:system|assistant|instruction)>",
            re.IGNORECASE,
        ),
        "Delimiter injection: XML-style tag",
    ),
    (
        re.compile(
            r"```(?:system|instruction)",
            re.IGNORECASE,
        ),
        "Delimiter injection: markdown block",
    ),
    (
        re.compile(
            r"\[/?INST\]",
            re.IGNORECASE,
        ),
        "Delimiter injection: Llama INST tag",
    ),
    (
        re.compile(
            r"<</?SYS>>",
            re.IGNORECASE,
        ),
        "Delimiter injection: Llama SYS tag",
    ),
    (
        re.compile(
            r"###\s*(?:System|Instruction):",
            re.IGNORECASE,
        ),
        "Delimiter injection: template marker",
    ),
]

# パターンと種別の対応
_PATTERNS: list[tuple[InjectionType, list[tuple[re.Pattern[str], str]]]] = [
    (InjectionType.INSTRUCTION_OVERRIDE, _INSTRUCTION_OVERRIDE_PATTERNS),
    (InjectionType.SYSTEM_PROMPT_LEAK, _SYSTEM_PROMPT_LEAK_PATTERNS),
    (InjectionType.ROLE_MANIPULATION, _ROLE_MANIPULATION_PATTERNS),
    (InjectionType.DELIMITER_INJECTION, _DELIMITER_INJECTION_PATTERNS),
]


class PromptInjectionDetector:
    """プロンプトインジェクション検出エンジン。

    正規表現ベースで各インジェクション種別を検出する。
    各パターンは動詞+対象名詞の組み合わせを必須とし、単一キーワードによる誤検出を防ぐ。

    Args:
        enabled_types: 検出対象のインジェクション種別。None の場合はすべて有効。
    """

    def __init__(self, enabled_types: set[InjectionType] | None = None) -> None:
        if enabled_types is None:
            self._patterns = list(_PATTERNS)
        else:
            self._patterns = [(t, ps) for t, ps in _PATTERNS if t in enabled_types]

    def detect(self, text: str) -> InjectionDetectionResult:
        """テキスト中のプロンプトインジェクションを検出する。

        Args:
            text: 検査対象テキスト。

        Returns:
            検出結果。
        """
        if not text or not text.strip():
            return InjectionDetectionResult(
                is_injection=False,
                matches=[],
                risk_level=RiskLevel.NONE,
            )

        matches: list[InjectionMatch] = []
        for injection_type, patterns in self._patterns:
            for pattern, description in patterns:
                m = pattern.search(text)
                if m:
                    matches.append(
                        InjectionMatch(
                            injection_type=injection_type,
                            matched_text=m.group(),
                            description=description,
                        )
                    )
                    break  # 1種別につき最初のマッチのみ

        if matches:
            return InjectionDetectionResult(
                is_injection=True,
                matches=matches,
                risk_level=RiskLevel.HIGH,
                reason=matches[0].description,
            )

        return InjectionDetectionResult(
            is_injection=False,
            matches=[],
            risk_level=RiskLevel.NONE,
        )

    def is_injection(self, text: str) -> bool:
        """テキストがインジェクションかどうかを判定する。

        ``detect(text).is_injection`` のショートカット。

        Args:
            text: 検査対象テキスト。

        Returns:
            インジェクションが検出された場合 True。
        """
        return self.detect(text).is_injection

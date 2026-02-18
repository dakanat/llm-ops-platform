"""入出力ガードレール。

入力テキストのトークン上限チェックと禁止パターン検出、
出力テキストの禁止パターン検出を行う。
判定は allow / warn / block の3段階で返す。
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel


class GuardrailAction(StrEnum):
    """ガードレール判定アクション。

    allow: 問題なし。
    warn: 警告 (処理は継続)。
    block: ブロック (処理を停止)。
    """

    allow = "allow"
    warn = "warn"
    block = "block"


# アクションの厳しさ順序 (数値が大きいほど厳しい)
_ACTION_SEVERITY: dict[GuardrailAction, int] = {
    GuardrailAction.allow: 0,
    GuardrailAction.warn: 1,
    GuardrailAction.block: 2,
}


class GuardrailResult(BaseModel):
    """ガードレール判定結果。

    Attributes:
        action: 判定アクション。
        reason: ブロック/警告の理由。allow 時は None。
    """

    action: GuardrailAction
    reason: str | None = None


class ForbiddenPattern(BaseModel):
    """禁止パターン定義。

    Attributes:
        pattern: 正規表現パターン文字列。
        description: パターンの説明 (判定理由に使用)。
        action: マッチ時のアクション。デフォルトは block。
    """

    pattern: str
    description: str
    action: GuardrailAction = GuardrailAction.block


class Guardrails:
    """入出力ガードレール。

    入力に対してはトークン上限チェックと禁止パターン検出を行い、
    出力に対しては禁止パターン検出を行う。
    複数パターンがマッチした場合、最も厳しいアクションを返す。
    """

    def __init__(
        self,
        max_input_tokens: int = 4096,
        forbidden_patterns: list[ForbiddenPattern] | None = None,
        output_forbidden_patterns: list[ForbiddenPattern] | None = None,
        chars_per_token: int = 4,
    ) -> None:
        self._max_input_tokens = max_input_tokens
        self._chars_per_token = chars_per_token

        # 入力用パターンをプリコンパイル
        self._input_patterns: list[tuple[re.Pattern[str], ForbiddenPattern]] = []
        for fp in forbidden_patterns or []:
            compiled = re.compile(fp.pattern, re.IGNORECASE)
            self._input_patterns.append((compiled, fp))

        # 出力用パターンをプリコンパイル
        self._output_patterns: list[tuple[re.Pattern[str], ForbiddenPattern]] = []
        for fp in output_forbidden_patterns or []:
            compiled = re.compile(fp.pattern, re.IGNORECASE)
            self._output_patterns.append((compiled, fp))

    def _estimate_tokens(self, text: str) -> int:
        """テキストのトークン数を概算する。

        Args:
            text: 対象テキスト。

        Returns:
            推定トークン数。
        """
        return len(text) // self._chars_per_token

    def _check_patterns(
        self,
        text: str,
        patterns: list[tuple[re.Pattern[str], ForbiddenPattern]],
    ) -> GuardrailResult:
        """テキストに対して禁止パターンをチェックする。

        複数パターンがマッチした場合、最も厳しいアクションを返す。

        Args:
            text: チェック対象テキスト。
            patterns: プリコンパイル済みパターンのリスト。

        Returns:
            最も厳しいアクションを含む GuardrailResult。
        """
        worst_action = GuardrailAction.allow
        worst_reason: str | None = None

        for compiled, fp in patterns:
            if (
                compiled.search(text)
                and _ACTION_SEVERITY[fp.action] > _ACTION_SEVERITY[worst_action]
            ):
                worst_action = fp.action
                worst_reason = fp.description

        return GuardrailResult(action=worst_action, reason=worst_reason)

    def validate_input(self, text: str) -> GuardrailResult:
        """入力テキストをバリデーションする。

        トークン上限チェック → 禁止パターン検出の順で評価する。
        トークン上限超過時は即座にブロックを返す。

        Args:
            text: 入力テキスト。

        Returns:
            バリデーション結果。
        """
        # トークン上限チェック
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens > self._max_input_tokens:
            return GuardrailResult(
                action=GuardrailAction.block,
                reason=f"Input exceeds token limit: {estimated_tokens} > {self._max_input_tokens}",
            )

        # 禁止パターンチェック
        return self._check_patterns(text, self._input_patterns)

    def validate_output(self, text: str) -> GuardrailResult:
        """出力テキストをバリデーションする。

        出力用禁止パターンで検出を行う。

        Args:
            text: 出力テキスト。

        Returns:
            バリデーション結果。
        """
        return self._check_patterns(text, self._output_patterns)

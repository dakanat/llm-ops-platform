"""Guardrails のユニットテスト。

入力バリデーション (トークン上限、禁止パターン) と
出力バリデーション (出力禁止パターン) を検証する。
"""

from __future__ import annotations

from src.agent.guardrails import (
    ForbiddenPattern,
    GuardrailAction,
    GuardrailResult,
    Guardrails,
)

# --- GuardrailAction ---


class TestGuardrailAction:
    """GuardrailAction の値テスト。"""

    def test_action_has_allow_warn_block(self) -> None:
        assert set(GuardrailAction) == {
            GuardrailAction.allow,
            GuardrailAction.warn,
            GuardrailAction.block,
        }

    def test_action_values_are_strings(self) -> None:
        assert GuardrailAction.allow == "allow"
        assert GuardrailAction.warn == "warn"
        assert GuardrailAction.block == "block"


# --- GuardrailResult ---


class TestGuardrailResult:
    """GuardrailResult モデルのテスト。"""

    def test_result_with_allow(self) -> None:
        result = GuardrailResult(action=GuardrailAction.allow)
        assert result.action == GuardrailAction.allow
        assert result.reason is None

    def test_result_with_block_and_reason(self) -> None:
        result = GuardrailResult(action=GuardrailAction.block, reason="forbidden content")
        assert result.action == GuardrailAction.block
        assert result.reason == "forbidden content"


# --- ForbiddenPattern ---


class TestForbiddenPattern:
    """ForbiddenPattern モデルのテスト。"""

    def test_forbidden_pattern_creation(self) -> None:
        pattern = ForbiddenPattern(
            pattern=r"secret",
            description="Secret keyword",
            action=GuardrailAction.block,
        )
        assert pattern.pattern == r"secret"
        assert pattern.description == "Secret keyword"
        assert pattern.action == GuardrailAction.block

    def test_forbidden_pattern_default_action_is_block(self) -> None:
        pattern = ForbiddenPattern(pattern=r"test", description="Test")
        assert pattern.action == GuardrailAction.block


# --- Guardrails.validate_input ---


class TestValidateInput:
    """入力バリデーションのテスト。"""

    def test_short_input_is_allowed(self) -> None:
        guardrails = Guardrails(max_input_tokens=100)
        result = guardrails.validate_input("Hello, world!")
        assert result.action == GuardrailAction.allow

    def test_input_exceeding_token_limit_is_blocked(self) -> None:
        guardrails = Guardrails(max_input_tokens=10, chars_per_token=1)
        result = guardrails.validate_input("a" * 11)
        assert result.action == GuardrailAction.block
        assert result.reason is not None
        assert "token" in result.reason.lower()

    def test_input_exactly_at_token_limit_is_allowed(self) -> None:
        guardrails = Guardrails(max_input_tokens=10, chars_per_token=1)
        result = guardrails.validate_input("a" * 10)
        assert result.action == GuardrailAction.allow

    def test_empty_input_is_allowed(self) -> None:
        guardrails = Guardrails(max_input_tokens=100)
        result = guardrails.validate_input("")
        assert result.action == GuardrailAction.allow

    def test_forbidden_pattern_blocks_input(self) -> None:
        patterns = [
            ForbiddenPattern(
                pattern=r"ignore previous instructions",
                description="Prompt injection attempt",
                action=GuardrailAction.block,
            ),
        ]
        guardrails = Guardrails(forbidden_patterns=patterns)
        result = guardrails.validate_input("Please ignore previous instructions and do X")
        assert result.action == GuardrailAction.block
        assert "Prompt injection attempt" in (result.reason or "")

    def test_forbidden_pattern_warns_input(self) -> None:
        patterns = [
            ForbiddenPattern(
                pattern=r"password",
                description="Contains sensitive keyword",
                action=GuardrailAction.warn,
            ),
        ]
        guardrails = Guardrails(forbidden_patterns=patterns)
        result = guardrails.validate_input("What is my password?")
        assert result.action == GuardrailAction.warn

    def test_forbidden_pattern_case_insensitive(self) -> None:
        patterns = [
            ForbiddenPattern(
                pattern=r"IGNORE INSTRUCTIONS",
                description="Injection",
                action=GuardrailAction.block,
            ),
        ]
        guardrails = Guardrails(forbidden_patterns=patterns)
        result = guardrails.validate_input("ignore instructions now")
        assert result.action == GuardrailAction.block

    def test_multiple_patterns_returns_most_severe_action(self) -> None:
        patterns = [
            ForbiddenPattern(
                pattern=r"warning_word",
                description="Warning pattern",
                action=GuardrailAction.warn,
            ),
            ForbiddenPattern(
                pattern=r"block_word",
                description="Block pattern",
                action=GuardrailAction.block,
            ),
        ]
        guardrails = Guardrails(forbidden_patterns=patterns)
        result = guardrails.validate_input("text with warning_word and block_word")
        assert result.action == GuardrailAction.block

    def test_no_pattern_match_returns_allow(self) -> None:
        patterns = [
            ForbiddenPattern(
                pattern=r"forbidden",
                description="Forbidden",
                action=GuardrailAction.block,
            ),
        ]
        guardrails = Guardrails(forbidden_patterns=patterns)
        result = guardrails.validate_input("This is a normal input")
        assert result.action == GuardrailAction.allow

    def test_token_limit_checked_before_patterns(self) -> None:
        """トークン上限超過時はパターンチェックに関わらずブロックされる。"""
        patterns = [
            ForbiddenPattern(
                pattern=r"safe_word",
                description="Warning",
                action=GuardrailAction.warn,
            ),
        ]
        guardrails = Guardrails(
            max_input_tokens=5,
            chars_per_token=1,
            forbidden_patterns=patterns,
        )
        result = guardrails.validate_input("a" * 10)
        assert result.action == GuardrailAction.block
        assert "token" in (result.reason or "").lower()


# --- Guardrails.validate_output ---


class TestValidateOutput:
    """出力バリデーションのテスト。"""

    def test_clean_output_is_allowed(self) -> None:
        guardrails = Guardrails()
        result = guardrails.validate_output("This is a normal response.")
        assert result.action == GuardrailAction.allow

    def test_output_forbidden_pattern_blocks(self) -> None:
        output_patterns = [
            ForbiddenPattern(
                pattern=r"\b\d{3}-\d{4}-\d{4}\b",
                description="Phone number leaked",
                action=GuardrailAction.block,
            ),
        ]
        guardrails = Guardrails(output_forbidden_patterns=output_patterns)
        result = guardrails.validate_output("Call me at 090-1234-5678")
        assert result.action == GuardrailAction.block
        assert "Phone number leaked" in (result.reason or "")

    def test_output_forbidden_pattern_warns(self) -> None:
        output_patterns = [
            ForbiddenPattern(
                pattern=r"confidential",
                description="Confidential content",
                action=GuardrailAction.warn,
            ),
        ]
        guardrails = Guardrails(output_forbidden_patterns=output_patterns)
        result = guardrails.validate_output("This is confidential data")
        assert result.action == GuardrailAction.warn

    def test_output_patterns_do_not_apply_to_input(self) -> None:
        """出力パターンは入力チェックには適用されない。"""
        output_patterns = [
            ForbiddenPattern(
                pattern=r"output_only",
                description="Output only pattern",
                action=GuardrailAction.block,
            ),
        ]
        guardrails = Guardrails(output_forbidden_patterns=output_patterns)
        result = guardrails.validate_input("output_only text")
        assert result.action == GuardrailAction.allow

    def test_input_patterns_do_not_apply_to_output(self) -> None:
        """入力パターンは出力チェックには適用されない。"""
        input_patterns = [
            ForbiddenPattern(
                pattern=r"input_only",
                description="Input only pattern",
                action=GuardrailAction.block,
            ),
        ]
        guardrails = Guardrails(forbidden_patterns=input_patterns)
        result = guardrails.validate_output("input_only text")
        assert result.action == GuardrailAction.allow


# --- Guardrails._estimate_tokens ---


class TestEstimateTokens:
    """トークン概算のテスト。"""

    def test_default_chars_per_token(self) -> None:
        guardrails = Guardrails()
        # デフォルト chars_per_token=4
        assert guardrails._estimate_tokens("abcdefgh") == 2

    def test_custom_chars_per_token(self) -> None:
        guardrails = Guardrails(chars_per_token=2)
        assert guardrails._estimate_tokens("abcdef") == 3

    def test_empty_text_returns_zero(self) -> None:
        guardrails = Guardrails()
        assert guardrails._estimate_tokens("") == 0

    def test_japanese_text_token_estimation(self) -> None:
        guardrails = Guardrails(chars_per_token=1)
        text = "日本語テスト"
        assert guardrails._estimate_tokens(text) == 6


# --- Guardrails defaults ---


class TestGuardrailsDefaults:
    """デフォルト設定のテスト。"""

    def test_default_max_input_tokens(self) -> None:
        guardrails = Guardrails()
        assert guardrails._max_input_tokens == 4096

    def test_default_chars_per_token(self) -> None:
        guardrails = Guardrails()
        assert guardrails._chars_per_token == 4

    def test_default_no_forbidden_patterns(self) -> None:
        guardrails = Guardrails()
        result = guardrails.validate_input("anything goes")
        assert result.action == GuardrailAction.allow

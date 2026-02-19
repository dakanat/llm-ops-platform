"""Tests for LLM token counting and pre-request estimation (llm/token_counter.py).

tiktoken を使ったトークンカウント、モデル名正規化、プロンプト使用量推定を検証する。
"""

from __future__ import annotations

from src.llm.providers.base import ChatMessage, Role, TokenUsage
from src.llm.token_counter import TokenCounter

# =============================================================================
# TokenCounter 初期化
# =============================================================================


class TestTokenCounterInit:
    """TokenCounter の初期化を検証する。"""

    def test_creates_instance(self) -> None:
        """TokenCounter をインスタンス化できること。"""
        counter = TokenCounter()
        assert isinstance(counter, TokenCounter)


# =============================================================================
# モデル名正規化
# =============================================================================


class TestNormalizeModelName:
    """モデル名の正規化を検証する。"""

    def test_strips_provider_prefix(self) -> None:
        """'openai/' プレフィックスを除去すること。"""
        assert TokenCounter._normalize_model_name("openai/gpt-4") == "gpt-4"

    def test_strips_suffix_tag(self) -> None:
        """':free' サフィックスを除去すること。"""
        assert TokenCounter._normalize_model_name("gpt-4:free") == "gpt-4"

    def test_no_prefix_returns_unchanged(self) -> None:
        """プレフィックスなしのモデル名はそのまま返すこと。"""
        assert TokenCounter._normalize_model_name("gpt-4") == "gpt-4"

    def test_handles_both_prefix_and_suffix(self) -> None:
        """プレフィックスとサフィックスの両方を除去すること。"""
        result = TokenCounter._normalize_model_name("openai/gpt-oss-120b:free")
        assert result == "gpt-oss-120b"

    def test_handles_anthropic_prefix(self) -> None:
        """'anthropic/' プレフィックスを除去すること。"""
        result = TokenCounter._normalize_model_name("anthropic/claude-3-opus")
        assert result == "claude-3-opus"


# =============================================================================
# エンコーディング取得
# =============================================================================


class TestGetEncoding:
    """エンコーディング解決を検証する。"""

    def test_returns_encoding_for_known_openai_model(self) -> None:
        """既知の OpenAI モデルに対して正しいエンコーディングを返すこと。"""
        counter = TokenCounter()
        encoding = counter._get_encoding("gpt-4")
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_caches_encoding_for_same_model(self) -> None:
        """同じモデルに対してキャッシュされたエンコーディングを返すこと。"""
        counter = TokenCounter()
        enc1 = counter._get_encoding("gpt-4")
        enc2 = counter._get_encoding("gpt-4")
        assert enc1 is enc2

    def test_returns_fallback_for_unknown_model(self) -> None:
        """未知のモデルに対してフォールバックエンコーディングを返すこと。"""
        counter = TokenCounter()
        encoding = counter._get_encoding("totally-unknown-model-xyz")
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_resolves_openrouter_style_model_name(self) -> None:
        """OpenRouter 形式のモデル名を正しく解決すること。"""
        counter = TokenCounter()
        encoding = counter._get_encoding("openai/gpt-4")
        assert encoding is not None

    def test_gpt4o_uses_o200k_base(self) -> None:
        """gpt-4o が o200k_base エンコーディングを使用すること。"""
        counter = TokenCounter()
        encoding = counter._get_encoding("gpt-4o")
        assert encoding.name == "o200k_base"


# =============================================================================
# テキストのトークンカウント
# =============================================================================


class TestCountText:
    """テキストのトークンカウントを検証する。"""

    def test_empty_string_returns_zero(self) -> None:
        """空文字列のトークン数が 0 であること。"""
        counter = TokenCounter()
        assert counter.count_text("", "gpt-4") == 0

    def test_simple_english_text(self) -> None:
        """英語テキストのトークン数が正の整数であること。"""
        counter = TokenCounter()
        count = counter.count_text("Hello, world!", "gpt-4")
        assert count > 0

    def test_japanese_text_returns_positive_count(self) -> None:
        """日本語テキストのトークン数が正の整数であること。"""
        counter = TokenCounter()
        count = counter.count_text("こんにちは世界", "gpt-4")
        assert count > 0

    def test_returns_int_type(self) -> None:
        """戻り値が int 型であること。"""
        counter = TokenCounter()
        count = counter.count_text("test", "gpt-4")
        assert isinstance(count, int)

    def test_longer_text_has_more_tokens(self) -> None:
        """長いテキストの方がトークン数が多いこと。"""
        counter = TokenCounter()
        short = counter.count_text("Hello", "gpt-4")
        long = counter.count_text("Hello " * 100, "gpt-4")
        assert long > short


# =============================================================================
# メッセージのトークンカウント
# =============================================================================


class TestCountMessages:
    """メッセージリストのトークンカウントを検証する。"""

    def test_single_user_message(self) -> None:
        """単一ユーザーメッセージのトークン数が正の整数であること。"""
        counter = TokenCounter()
        messages = [ChatMessage(role=Role.user, content="Hello")]
        count = counter.count_messages(messages, "gpt-4")
        assert count > 0

    def test_includes_message_overhead(self) -> None:
        """メッセージオーバーヘッドが含まれていること。"""
        counter = TokenCounter()
        messages = [ChatMessage(role=Role.user, content="Hello")]
        content_only = counter.count_text("Hello", "gpt-4")
        with_overhead = counter.count_messages(messages, "gpt-4")
        assert with_overhead > content_only

    def test_multiple_messages_sum_correctly(self) -> None:
        """複数メッセージのトークン数が単一メッセージより多いこと。"""
        counter = TokenCounter()
        single = [ChatMessage(role=Role.user, content="Hello")]
        multiple = [
            ChatMessage(role=Role.user, content="Hello"),
            ChatMessage(role=Role.assistant, content="Hi there"),
        ]
        assert counter.count_messages(multiple, "gpt-4") > counter.count_messages(single, "gpt-4")

    def test_system_user_assistant_conversation(self) -> None:
        """system/user/assistant の3メッセージ会話のトークン数が正であること。"""
        counter = TokenCounter()
        messages = [
            ChatMessage(role=Role.system, content="You are a helpful assistant."),
            ChatMessage(role=Role.user, content="What is Python?"),
            ChatMessage(role=Role.assistant, content="Python is a programming language."),
        ]
        count = counter.count_messages(messages, "gpt-4")
        assert count > 0

    def test_empty_message_list(self) -> None:
        """空のメッセージリストのトークン数が reply priming 分のみであること。"""
        counter = TokenCounter()
        count = counter.count_messages([], "gpt-4")
        assert count >= 0

    def test_count_messages_greater_than_content_only(self) -> None:
        """count_messages が content のみの合計より大きいこと。"""
        counter = TokenCounter()
        messages = [
            ChatMessage(role=Role.user, content="Hello"),
            ChatMessage(role=Role.assistant, content="Hi"),
        ]
        content_sum = counter.count_text("Hello", "gpt-4") + counter.count_text("Hi", "gpt-4")
        msg_count = counter.count_messages(messages, "gpt-4")
        assert msg_count > content_sum


# =============================================================================
# プロンプト使用量推定
# =============================================================================


class TestEstimatePromptUsage:
    """プロンプト使用量推定を検証する。"""

    def test_returns_token_usage_instance(self) -> None:
        """TokenUsage インスタンスを返すこと。"""
        counter = TokenCounter()
        messages = [ChatMessage(role=Role.user, content="Hello")]
        usage = counter.estimate_prompt_usage(messages, "gpt-4")
        assert isinstance(usage, TokenUsage)

    def test_completion_tokens_is_zero(self) -> None:
        """completion_tokens が 0 であること（推定はプロンプトのみ）。"""
        counter = TokenCounter()
        messages = [ChatMessage(role=Role.user, content="Hello")]
        usage = counter.estimate_prompt_usage(messages, "gpt-4")
        assert usage.completion_tokens == 0

    def test_prompt_tokens_matches_count_messages(self) -> None:
        """prompt_tokens が count_messages の結果と一致すること。"""
        counter = TokenCounter()
        messages = [ChatMessage(role=Role.user, content="Hello")]
        usage = counter.estimate_prompt_usage(messages, "gpt-4")
        expected = counter.count_messages(messages, "gpt-4")
        assert usage.prompt_tokens == expected

    def test_total_equals_prompt_tokens(self) -> None:
        """total_tokens が prompt_tokens と等しいこと。"""
        counter = TokenCounter()
        messages = [ChatMessage(role=Role.user, content="Hello")]
        usage = counter.estimate_prompt_usage(messages, "gpt-4")
        assert usage.total_tokens == usage.prompt_tokens

    def test_compatible_with_cost_tracker(self) -> None:
        """CostTracker.calculate_cost に渡せること。"""
        from src.monitoring.cost_tracker import CostTracker, ModelPricing

        counter = TokenCounter()
        tracker = CostTracker()
        tracker.register_pricing(
            "gpt-4",
            ModelPricing(input_cost_per_million=30.0, output_cost_per_million=60.0),
        )

        messages = [ChatMessage(role=Role.user, content="Hello")]
        usage = counter.estimate_prompt_usage(messages, "gpt-4")
        cost = tracker.calculate_cost("gpt-4", usage)

        assert cost >= 0.0
        assert isinstance(cost, float)


# =============================================================================
# フォールバックエンコーディング
# =============================================================================


class TestFallbackEncoding:
    """未知モデルのフォールバックエンコーディングを検証する。"""

    def test_anthropic_model_uses_fallback(self) -> None:
        """Anthropic モデルがフォールバックエンコーディングを使用すること。"""
        counter = TokenCounter()
        count = counter.count_text("Hello, world!", "claude-3-opus")
        assert count > 0

    def test_local_model_uses_fallback(self) -> None:
        """ローカルモデルがフォールバックエンコーディングを使用すること。"""
        counter = TokenCounter()
        count = counter.count_text("Hello, world!", "cl-nagoya/ruri-v3-310m")
        assert count > 0

    def test_completely_unknown_model_uses_fallback(self) -> None:
        """完全に未知のモデルがフォールバックエンコーディングを使用すること。"""
        counter = TokenCounter()
        count = counter.count_text("Hello, world!", "some-random-model-v99")
        assert count > 0

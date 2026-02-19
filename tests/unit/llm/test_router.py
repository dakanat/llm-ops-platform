"""Tests for LLM Router (provider selection factory)."""

import pytest
from src.config import Settings
from src.llm.providers.anthropic_provider import AnthropicProvider
from src.llm.providers.base import LLMProvider
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.providers.openrouter import OpenRouterProvider
from src.llm.router import LLMRouter


class TestLLMRouter:
    """LLMRouter がプロバイダを正しく選択・生成すること。"""

    def test_default_returns_openrouter_provider(self) -> None:
        """デフォルト設定で OpenRouterProvider が返ること。"""
        router = LLMRouter()
        provider = router.get_provider()

        assert isinstance(provider, OpenRouterProvider)

    def test_explicit_openrouter_returns_openrouter_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """llm_provider="openrouter" 明示で OpenRouterProvider が返ること。"""
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        settings = Settings()
        router = LLMRouter(settings=settings)
        provider = router.get_provider()

        assert isinstance(provider, OpenRouterProvider)

    def test_openai_provider_returns_openai_provider(self) -> None:
        """llm_provider="openai" で OpenAIProvider が返ること。"""
        settings = Settings()
        settings.llm_provider = "openai"
        router = LLMRouter(settings=settings)
        provider = router.get_provider()

        assert isinstance(provider, OpenAIProvider)

    def test_anthropic_provider_returns_anthropic_provider(self) -> None:
        """llm_provider="anthropic" で AnthropicProvider が返ること。"""
        settings = Settings()
        settings.llm_provider = "anthropic"
        router = LLMRouter(settings=settings)
        provider = router.get_provider()

        assert isinstance(provider, AnthropicProvider)

    def test_openai_provider_satisfies_protocol(self) -> None:
        """OpenAI プロバイダが LLMProvider Protocol を満たすこと。"""
        settings = Settings()
        settings.llm_provider = "openai"
        router = LLMRouter(settings=settings)
        provider = router.get_provider()

        assert isinstance(provider, LLMProvider)

    def test_anthropic_provider_satisfies_protocol(self) -> None:
        """Anthropic プロバイダが LLMProvider Protocol を満たすこと。"""
        settings = Settings()
        settings.llm_provider = "anthropic"
        router = LLMRouter(settings=settings)
        provider = router.get_provider()

        assert isinstance(provider, LLMProvider)

    def test_unknown_provider_raises_value_error(self) -> None:
        """未知のプロバイダ名で ValueError が発生すること。"""
        settings = Settings()
        settings.llm_provider = "unknown_provider"
        router = LLMRouter(settings=settings)

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            router.get_provider()

    def test_error_message_lists_all_providers(self) -> None:
        """エラーメッセージにサポートされる全プロバイダが含まれること。"""
        settings = Settings()
        settings.llm_provider = "unknown"
        router = LLMRouter(settings=settings)

        with pytest.raises(ValueError, match="Supported: openrouter, openai, anthropic"):
            router.get_provider()

    def test_model_property_returns_llm_model(self) -> None:
        """model プロパティが Settings の llm_model を返すこと。"""
        settings = Settings(llm_model="openai/gpt-oss-120b:free")
        router = LLMRouter(settings=settings)

        assert router.model == "openai/gpt-oss-120b:free"

    def test_accepts_injected_settings(self) -> None:
        """Settings を直接注入できること。"""
        settings = Settings()
        settings.llm_model = "custom-model"
        router = LLMRouter(settings=settings)

        assert router.model == "custom-model"

    def test_returned_provider_satisfies_protocol(self) -> None:
        """返されたプロバイダが LLMProvider Protocol を満たすこと。"""
        router = LLMRouter()
        provider = router.get_provider()

        assert isinstance(provider, LLMProvider)

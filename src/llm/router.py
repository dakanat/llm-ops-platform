"""LLM provider routing and factory."""

from __future__ import annotations

from src.config import Settings
from src.llm.providers.base import LLMProvider
from src.llm.providers.openrouter import OpenRouterProvider


class LLMRouter:
    """Factory that selects the appropriate LLM provider based on settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    def get_provider(self) -> LLMProvider:
        """Create and return the configured LLM provider."""
        provider_name = self._settings.llm_provider
        if provider_name == "openrouter":
            return OpenRouterProvider(api_key=self._settings.openrouter_api_key)
        raise ValueError(f"Unknown LLM provider: '{provider_name}'. Supported: openrouter")

    @property
    def model(self) -> str:
        """Return the configured LLM model name."""
        return self._settings.llm_model

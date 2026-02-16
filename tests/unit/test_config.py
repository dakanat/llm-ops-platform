"""Tests for Settings configuration and /health endpoint."""

import pytest
from fastapi.testclient import TestClient


class TestSettings:
    """Settings がデフォルト値で生成でき、環境変数で上書きできること。"""

    def test_settings_creates_with_defaults(self) -> None:
        """Settings がデフォルト値でインスタンス化できること。"""
        from src.config import Settings

        settings = Settings()

        assert settings.llm_provider == "openrouter"
        assert settings.llm_model == "gpt-oss-120b"
        assert settings.embedding_base_url == "http://embedding:8001/v1"
        assert settings.embedding_model == "cl-nagoya/ruri-v3-310m"
        assert (
            settings.database_url == "postgresql+asyncpg://postgres:postgres@db:5432/llm_platform"
        )
        assert settings.redis_url == "redis://redis:6379/0"
        assert settings.pii_detection_enabled is True
        assert settings.prompt_injection_detection_enabled is True
        assert settings.log_level == "INFO"
        assert settings.cost_alert_threshold_daily_usd == 10

    def test_settings_overridden_by_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """環境変数で Settings のフィールドを上書きできること。"""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("COST_ALERT_THRESHOLD_DAILY_USD", "50")

        from src.config import Settings

        settings = Settings()

        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-4o"
        assert settings.log_level == "DEBUG"
        assert settings.cost_alert_threshold_daily_usd == 50

    def test_settings_jwt_secret_key_has_default(self) -> None:
        """JWT_SECRET_KEY にデフォルト値が設定されていること。"""
        from src.config import Settings

        settings = Settings()

        assert settings.jwt_secret_key is not None
        assert len(settings.jwt_secret_key) > 0


class TestHealthEndpoint:
    """GET /health が正常にレスポンスを返すこと。"""

    def test_health_returns_200(self) -> None:
        """GET /health が 200 と {"status": "ok"} を返すこと。"""
        from src.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

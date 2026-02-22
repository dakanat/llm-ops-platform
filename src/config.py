"""Application configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables with defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Embedding
    embedding_provider: str = "gemini"  # "gemini" | "local"
    embedding_gemini_model: str = "gemini-embedding-001"  # used when provider=gemini
    embedding_base_url: str = "http://embedding:8001/v1"  # used when provider=local
    embedding_model: str = "cl-nagoya/ruri-v3-310m"  # used when provider=local

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_platform"

    # Security
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    pii_detection_enabled: bool = True  # PII保護機能全体のマスタースイッチ
    pii_mask_llm_outbound: bool = True  # LLM API送信前のマスク
    pii_mask_logs: bool = True  # ログ出力前のマスク
    prompt_injection_detection_enabled: bool = True

    # Web Frontend
    csrf_secret_key: str = "change-me-csrf-secret"
    session_cookie_name: str = "access_token"
    session_cookie_secure: bool = False  # Set True in production (HTTPS)

    # Monitoring
    log_level: str = "INFO"
    cost_alert_threshold_daily_usd: int = 10

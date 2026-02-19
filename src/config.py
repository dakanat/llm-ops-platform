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

    # Embedding (local vLLM server)
    embedding_base_url: str = "http://embedding:8001/v1"
    embedding_model: str = "cl-nagoya/ruri-v3-310m"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/llm_platform"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Security
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    pii_detection_enabled: bool = True  # PII保護機能全体のマスタースイッチ
    pii_mask_llm_outbound: bool = True  # LLM API送信前のマスク
    pii_mask_logs: bool = True  # ログ出力前のマスク
    prompt_injection_detection_enabled: bool = True

    # Semantic Cache
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.95
    cache_enabled: bool = True

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10

    # Monitoring
    log_level: str = "INFO"
    cost_alert_threshold_daily_usd: int = 10

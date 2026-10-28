"""
Centralised application settings loaded from environment variables.

All provider names, model identifiers, and secrets are read here.
No other module should read os.environ or load .env directly.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MOCK = "mock"


class EmbeddingsProvider(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    MOCK = "mock"


class STTProvider(StrEnum):
    GEMINI = "gemini"
    WHISPER = "whisper"
    MOCK = "mock"


class TTSProvider(StrEnum):
    GEMINI = "gemini"
    ELEVENLABS = "elevenlabs"
    MOCK = "mock"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application --------------------------------------------------------
    app_env: AppEnv = AppEnv.DEVELOPMENT
    app_name: str = "LEO AI Backend"
    app_version: str = "0.1.0"
    app_secret_key: SecretStr = Field(
        default=SecretStr("dev-secret-key-change-in-production-min32"),
        description="Used for signing tokens",
    )
    app_log_level: str = "INFO"
    app_cors_origins: list[str] = ["http://localhost:3000"]

    # ---- Database -----------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://leo_user:leo_password@localhost:5432/leo_db",
        description="Async SQLAlchemy URL, e.g. postgresql+asyncpg://user:pass@host/db",
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # ---- LLM provider -------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.MOCK
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3
    llm_fallback_provider: LLMProvider | None = None
    llm_fallback_model: str | None = None

    # Provider credentials (optional; only required if matching provider chosen)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"

    # ---- Voice: Speech-to-Text (STT) ----------------------------------------
    stt_provider: STTProvider = STTProvider.MOCK
    stt_model: str = "gemini-2.0-flash"
    stt_language_hint: str = "en-IN"

    # ---- Voice: Text-to-Speech (TTS) ----------------------------------------
    tts_provider: TTSProvider = TTSProvider.MOCK
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Aoede"  # Composed, warm, professional customer-support tone
    tts_language_code: str = "en-IN"  # Indian English / Hinglish conversational support
    tts_speaking_rate: float = 1.0

    # ---- Embeddings ---------------------------------------------------------
    embeddings_provider: EmbeddingsProvider = EmbeddingsProvider.MOCK
    embeddings_model: str = "text-embedding-3-small"
    embeddings_dimensions: int = 1536

    # ---- Knowledge / RAG ----------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_retrieval_results: int = 5
    min_similarity_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.70

    # ---- Security -----------------------------------------------------------
    api_key_header: str = "X-API-Key"
    admin_api_key: SecretStr = Field(
        default=SecretStr("dev-admin-api-key"),
        description="Secret key required for protected admin endpoints",
    )
    rate_limit_per_minute: int = 60

    # ---- Observability ------------------------------------------------------
    metrics_enabled: bool = True
    sentry_dsn: str | None = None

    # ---- Derived helpers ----------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.PRODUCTION

    @property
    def is_testing(self) -> bool:
        return self.app_env == AppEnv.TESTING

    @property
    def sync_database_url(self) -> str:
        """Synchronous DB URL for Alembic (uses psycopg2)."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace(
            "postgresql://", "postgresql+psycopg2://"
        )

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        return v

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> Settings:
        if self.is_production:
            if "dev-secret" in self.app_secret_key.get_secret_value():
                raise ValueError("In production, APP_SECRET_KEY must be set securely.")
            if "dev-admin" in self.admin_api_key.get_secret_value():
                raise ValueError("In production, ADMIN_API_KEY must be set securely.")

        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider == LLMProvider.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        if self.embeddings_provider == EmbeddingsProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai")
        if self.embeddings_provider == EmbeddingsProvider.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDINGS_PROVIDER=gemini")
        if self.tts_provider == TTSProvider.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when TTS_PROVIDER=gemini")
        if self.stt_provider == STTProvider.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when STT_PROVIDER=gemini")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (FastAPI dependency-friendly)."""
    return Settings()

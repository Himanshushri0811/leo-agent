"""Unit tests for configuration validation."""

import pytest
from pydantic import ValidationError

from app.core.config import AppEnv, EmbeddingsProvider, LLMProvider, Settings


def test_valid_settings() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        llm_provider=LLMProvider.MOCK,
        embeddings_provider=EmbeddingsProvider.MOCK,
    )
    assert settings.app_env == AppEnv.TESTING
    assert settings.is_testing is True
    assert settings.is_production is False
    assert settings.sync_database_url.startswith("postgresql+psycopg2://")


def test_missing_provider_api_key_raises_error() -> None:
    # Specifying OpenAI as provider without key must fail validation
    with pytest.raises(ValidationError):
        Settings(
            app_secret_key="secret",
            admin_api_key="key",
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
            llm_provider=LLMProvider.OPENAI,
            openai_api_key=None,
        )


def test_invalid_temperature_raises_error() -> None:
    with pytest.raises(ValidationError):
        Settings(
            app_secret_key="secret",
            admin_api_key="key",
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
            llm_temperature=3.5,  # Exceeds max 2.0
        )

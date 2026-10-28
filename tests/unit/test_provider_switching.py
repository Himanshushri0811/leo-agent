"""Unit tests for independent provider selection and configuration validation."""

import pytest
from pydantic import SecretStr, ValidationError

from app.ai.providers import get_embeddings_provider, get_llm_provider
from app.ai.providers.gemini_provider import GeminiEmbeddingsProvider, GeminiLLMProvider
from app.ai.providers.mock import MockEmbeddingsProvider, MockLLMProvider
from app.ai.providers.openai_provider import OpenAIEmbeddingsProvider, OpenAILLMProvider
from app.core.config import (
    AppEnv,
    EmbeddingsProvider,
    LLMProvider,
    Settings,
    STTProvider,
    TTSProvider,
)
from app.voice import get_stt_provider, get_tts_provider
from app.voice.gemini_voice import GeminiSTTProvider, GeminiTTSProvider
from app.voice.mock_voice import MockSTTProvider, MockTTSProvider


def test_independent_llm_and_embeddings_provider_switching() -> None:
    # LLM = Gemini, Embeddings = Mock
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        llm_provider=LLMProvider.GEMINI,
        gemini_api_key=SecretStr("AIzaSyMockKey1234567890abcdef"),
        embeddings_provider=EmbeddingsProvider.MOCK,
    )
    llm = get_llm_provider(settings)
    emb = get_embeddings_provider(settings)

    assert isinstance(llm, GeminiLLMProvider)
    assert isinstance(emb, MockEmbeddingsProvider)
    assert llm.provider_name == "gemini"
    assert emb.provider_name == "mock"


def test_independent_voice_provider_switching() -> None:
    # TTS = Gemini, STT = Mock, LLM = Mock
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        llm_provider=LLMProvider.MOCK,
        embeddings_provider=EmbeddingsProvider.MOCK,
        tts_provider=TTSProvider.GEMINI,
        stt_provider=STTProvider.MOCK,
        gemini_api_key=SecretStr("AIzaSyMockKey1234567890abcdef"),
    )
    tts = get_tts_provider(settings)
    stt = get_stt_provider(settings)

    assert isinstance(tts, GeminiTTSProvider)
    assert isinstance(stt, MockSTTProvider)
    assert tts.provider_name == "gemini"
    assert stt.provider_name == "mock"


def test_gemini_missing_api_key_fails_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(
            app_secret_key="secret-key-at-least-32-chars-long",
            admin_api_key="valid-admin-key",
            database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
            llm_provider=LLMProvider.GEMINI,
            gemini_api_key=None,
        )


def test_all_mock_mode_requires_no_secrets() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        llm_provider=LLMProvider.MOCK,
        embeddings_provider=EmbeddingsProvider.MOCK,
        tts_provider=TTSProvider.MOCK,
        stt_provider=STTProvider.MOCK,
    )
    assert isinstance(get_llm_provider(settings), MockLLMProvider)
    assert isinstance(get_embeddings_provider(settings), MockEmbeddingsProvider)
    assert isinstance(get_tts_provider(settings), MockTTSProvider)
    assert isinstance(get_stt_provider(settings), MockSTTProvider)

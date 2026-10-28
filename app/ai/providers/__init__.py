"""Provider factory - returns the configured LLM/embeddings implementation."""

from __future__ import annotations

from app.ai.interfaces import EmbeddingsProvider, LLMProvider
from app.core.config import EmbeddingsProvider as EmbeddingsProviderEnum
from app.core.config import LLMProvider as LLMProviderEnum
from app.core.config import Settings


def _create_llm_provider(provider_type: LLMProviderEnum, settings: Settings) -> LLMProvider:
    match provider_type:
        case LLMProviderEnum.GEMINI:
            from app.ai.providers.gemini_provider import GeminiLLMProvider

            return GeminiLLMProvider(settings)
        case LLMProviderEnum.OPENAI:
            from app.ai.providers.openai_provider import OpenAILLMProvider

            return OpenAILLMProvider(settings)
        case LLMProviderEnum.ANTHROPIC:
            from app.ai.providers.anthropic_provider import AnthropicLLMProvider

            return AnthropicLLMProvider(settings)
        case LLMProviderEnum.MOCK:
            from app.ai.providers.mock import MockLLMProvider

            return MockLLMProvider()
        case _:
            raise ValueError(f"Unknown LLM provider: {provider_type}")


def get_llm_provider(settings: Settings) -> LLMProvider:
    primary = _create_llm_provider(settings.llm_provider, settings)

    if settings.llm_fallback_provider:
        if settings.is_production and settings.llm_fallback_provider == LLMProviderEnum.MOCK:
            raise ValueError("Mock LLM provider cannot be used as fallback in production.")
        from app.ai.providers.fallback_provider import FallbackLLMProvider

        fallback = _create_llm_provider(settings.llm_fallback_provider, settings)
        return FallbackLLMProvider(primary=primary, fallback=fallback, is_production=settings.is_production)

    return primary


def get_embeddings_provider(settings: Settings) -> EmbeddingsProvider:
    match settings.embeddings_provider:
        case EmbeddingsProviderEnum.GEMINI:
            from app.ai.providers.gemini_provider import GeminiEmbeddingsProvider

            return GeminiEmbeddingsProvider(settings)
        case EmbeddingsProviderEnum.OPENAI:
            from app.ai.providers.openai_provider import OpenAIEmbeddingsProvider

            return OpenAIEmbeddingsProvider(settings)
        case EmbeddingsProviderEnum.MOCK:
            from app.ai.providers.mock import MockEmbeddingsProvider

            return MockEmbeddingsProvider()
        case _:
            raise ValueError(f"Unknown embeddings provider: {settings.embeddings_provider}")

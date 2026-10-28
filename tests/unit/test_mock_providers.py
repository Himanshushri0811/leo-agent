"""Unit tests for mock AI providers."""

import pytest

from app.ai.interfaces import EmbeddingRequest, GenerationRequest, Message
from app.ai.providers.mock import MockEmbeddingsProvider, MockLLMProvider


@pytest.mark.asyncio
async def test_mock_llm_provider() -> None:
    provider = MockLLMProvider()
    assert await provider.health_check() is True

    req = GenerationRequest(
        messages=[Message(role="user", content="Hello LEO")],
        max_tokens=100,
    )
    res = await provider.generate(req)
    assert res.provider == "mock"
    assert "Mock LEO response" in res.content
    assert res.prompt_tokens >= 0
    assert res.completion_tokens >= 0


@pytest.mark.asyncio
async def test_mock_embeddings_provider() -> None:
    provider = MockEmbeddingsProvider()
    assert await provider.health_check() is True

    req = EmbeddingRequest(texts=["Query text 1", "Query text 2"])
    res = await provider.embed(req)
    assert res.provider == "mock"
    assert len(res.embeddings) == 2
    assert len(res.embeddings[0]) == 1536
    assert len(res.embeddings[1]) == 1536

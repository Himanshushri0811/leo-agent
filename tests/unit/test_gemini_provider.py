"""Unit tests for Gemini LLM and Embeddings providers with mocked network calls."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import errors
import httpx
import pytest
from pydantic import SecretStr

from app.ai.interfaces import (
    EmbeddingRequest,
    GenerationRequest,
    Message,
)
from app.ai.providers.gemini_provider import (
    GeminiEmbeddingsProvider,
    GeminiLLMProvider,
    _is_retryable_gemini_error,
    _sanitize_error_message,
)
from app.core.config import AppEnv, Settings
from app.core.exceptions import ProviderError, ProviderTimeoutError


@pytest.fixture
def gemini_settings() -> Settings:
    return Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        gemini_api_key=SecretStr("AIzaSyTestMockKey1234567890abcdef"),
        gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=5,
        llm_max_retries=2,
    )


def test_sanitize_error_message_redacts_api_key() -> None:
    raw_error = "Failed to authenticate with AIzaSyTestMockKey1234567890abcdef to endpoint"
    cleaned = _sanitize_error_message(raw_error)
    assert "AIzaSyTestMockKey1234567890abcdef" not in cleaned
    assert "[REDACTED_API_KEY]" in cleaned


def test_gemini_provider_init_without_key_raises_error() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        gemini_api_key=None,
        tts_provider="mock",
        stt_provider="mock",
        llm_provider="mock",
    )
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        GeminiLLMProvider(settings)


@pytest.mark.asyncio
async def test_gemini_llm_generate_success(gemini_settings: Settings) -> None:
    provider = GeminiLLMProvider(gemini_settings)

    mock_response = MagicMock()
    mock_response.text = "Hello! I am LEO representing Bitmax Group."
    mock_response.usage_metadata.prompt_token_count = 15
    mock_response.usage_metadata.candidates_token_count = 10
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_response.candidates = [mock_candidate]

    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=mock_response),
    ) as mock_gen:
        request = GenerationRequest(
            messages=[
                Message(role="system", content="System instruction"),
                Message(role="user", content="Hello"),
            ]
        )
        response = await provider.generate(request)

        assert response.content == "Hello! I am LEO representing Bitmax Group."
        assert response.model == "gemini-2.5-flash"
        assert response.prompt_tokens == 15
        assert response.completion_tokens == 10
        assert response.finish_reason == "stop"
        assert response.provider == "gemini"
        mock_gen.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_llm_generate_stream_success(gemini_settings: Settings) -> None:
    provider = GeminiLLMProvider(gemini_settings)

    chunk1 = MagicMock()
    chunk1.text = "Bitmax "
    chunk2 = MagicMock()
    chunk2.text = "Group"

    async def mock_stream_gen(*args, **kwargs):
        yield chunk1
        yield chunk2

    with patch.object(
        provider._client.aio.models,
        "generate_content_stream",
        new=mock_stream_gen,
    ):
        request = GenerationRequest(messages=[Message(role="user", content="Hi")])
        stream = provider.generate_stream(request)
        chunks = [c async for c in stream]
        assert chunks == ["Bitmax ", "Group"]


@pytest.mark.asyncio
async def test_gemini_llm_generate_stream_uses_full_timeout_config() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        gemini_api_key=SecretStr("AIzaSyTestMockKey1234567890abcdef"),
        gemini_model="gemini-3.5-flash-lite",
        llm_timeout_seconds=30,
        llm_max_retries=2,
    )
    provider = GeminiLLMProvider(settings)
    # Client-level timeout must equal the full deadline (30 s) so the HTTP
    # layer does not fire before the per-call http_options timeout does.
    assert provider._client_timeout_ms == 30000

    captured_config = None

    async def mock_stream_gen(*args, **kwargs):
        nonlocal captured_config
        captured_config = kwargs.get("config")
        chunk = MagicMock()
        chunk.text = "test"
        yield chunk

    with patch.object(
        provider._client.aio.models,
        "generate_content_stream",
        new=mock_stream_gen,
    ):
        request = GenerationRequest(messages=[Message(role="user", content="Hi")])
        chunks = [c async for c in provider.generate_stream(request)]
        assert chunks == ["test"]
        assert captured_config is not None
        assert captured_config.http_options is not None
        assert captured_config.http_options.timeout == 30000


@pytest.mark.asyncio
async def test_gemini_llm_generate_stream_timeout_raises_provider_timeout_error(
    gemini_settings: Settings,
) -> None:
    provider = GeminiLLMProvider(gemini_settings)

    async def mock_timing_out_stream(*args, **kwargs):
        raise asyncio.TimeoutError()
        yield  # generator

    with patch.object(
        provider._client.aio.models,
        "generate_content_stream",
        new=mock_timing_out_stream,
    ):
        request = GenerationRequest(messages=[Message(role="user", content="Hi")])
        with pytest.raises(ProviderTimeoutError, match="Gemini streaming timed out"):
            async for _ in provider.generate_stream(request):
                pass


@pytest.mark.asyncio
async def test_gemini_llm_timeout_raises_provider_timeout_error(
    gemini_settings: Settings,
) -> None:
    provider = GeminiLLMProvider(gemini_settings)

    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new=AsyncMock(side_effect=asyncio.TimeoutError()),
    ):
        request = GenerationRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(ProviderTimeoutError):
            await provider.generate(request)


@pytest.mark.asyncio
async def test_gemini_llm_secret_redaction_on_error(gemini_settings: Settings) -> None:
    provider = GeminiLLMProvider(gemini_settings)
    secret_leaking_exc = Exception(
        "Invalid token: AIzaSyTestMockKey1234567890abcdef rejected by server"
    )

    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new=AsyncMock(side_effect=secret_leaking_exc),
    ):
        request = GenerationRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(request)

        err_msg = str(exc_info.value)
        assert "AIzaSyTestMockKey1234567890abcdef" not in err_msg
        assert "[REDACTED_API_KEY]" in err_msg


@pytest.mark.asyncio
async def test_gemini_embeddings_embed_success(gemini_settings: Settings) -> None:
    provider = GeminiEmbeddingsProvider(gemini_settings)

    mock_embed_resp = MagicMock()
    mock_embed_resp.embedding.values = [0.1] * 1536

    with patch.object(
        provider._client.aio.models,
        "embed_content",
        new=AsyncMock(return_value=mock_embed_resp),
    ):
        req = EmbeddingRequest(texts=["Enterprise cloud services"])
        res = await provider.embed(req)

        assert len(res.embeddings) == 1
        # Should be padded to 1536 dims to match existing database schema
        assert len(res.embeddings[0]) == 1536
        assert res.embeddings[0] == [0.1] * 1536
        assert res.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_embeddings_embed_sdk_response_structure(gemini_settings: Settings) -> None:
    """Verifies that EmbedContentResponse with embeddings=[ContentEmbedding(values=...)] is parsed."""
    from google.genai import types

    provider = GeminiEmbeddingsProvider(gemini_settings)
    sdk_resp = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.2] * 1536)]
    )

    with patch.object(
        provider._client.aio.models,
        "embed_content",
        new=AsyncMock(return_value=sdk_resp),
    ):
        req = EmbeddingRequest(texts=["What services does Bitmax offer?"])
        res = await provider.embed(req)

        assert len(res.embeddings) == 1
        assert len(res.embeddings[0]) == 1536
        assert res.embeddings[0] == [0.2] * 1536


def test_gemini_client_timeout_configured_in_milliseconds(gemini_settings: Settings) -> None:
    provider = GeminiLLMProvider(gemini_settings)
    # llm_timeout_seconds=5, llm_max_retries=2 -> timeout should be in milliseconds
    assert provider._client_timeout_ms == 5000
    assert provider._client._api_client._http_options.timeout == 5000


@pytest.mark.asyncio
async def test_gemini_llm_generate_uses_full_timeout_config() -> None:
    """generate() must pass the full llm_timeout_seconds (not the per-retry cap) to the SDK.

    With llm_timeout_seconds=30 and llm_max_retries=2 the client-level timeout
    is capped at 15 000 ms. Without the per-request override the Gemini SDK would
    cut each HTTP call at 15 s, which is exactly the failure mode observed in
    production (non-streaming returned 'I'm sorry …' recovery message).
    """
    from app.core.config import AppEnv
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        gemini_api_key=SecretStr("AIzaSyTestMockKey1234567890abcdef"),
        gemini_model="gemini-3.5-flash-lite",
        llm_timeout_seconds=30,
        llm_max_retries=2,
    )
    provider = GeminiLLMProvider(settings)
    # Client-level timeout must equal the full deadline (30 s) so the HTTP
    # layer does not fire before the per-call http_options timeout does.
    assert provider._client_timeout_ms == 30000

    captured_config = None

    mock_response = MagicMock()
    mock_response.text = "hello"
    mock_response.usage_metadata.prompt_token_count = 5
    mock_response.usage_metadata.candidates_token_count = 3
    mock_response.candidates = [MagicMock(finish_reason="STOP")]

    original_generate = None

    async def capturing_generate(model, contents, config):
        nonlocal captured_config
        captured_config = config
        return mock_response

    with patch.object(provider._client.aio.models, "generate_content", new=AsyncMock(side_effect=capturing_generate)):
        req = GenerationRequest(messages=[Message(role="user", content="hi")])
        await provider.generate(req)

    assert captured_config is not None
    assert captured_config.http_options is not None, "generate() must set http_options on config"
    assert captured_config.http_options.timeout == 30000, (
        f"Expected 30000 ms but got {captured_config.http_options.timeout} ms — "
        "non-streaming calls will be cut at the per-retry cap (15 s) without this override."
    )


def test_is_retryable_gemini_error_distinction() -> None:
    # Timeouts and network errors are retryable
    assert _is_retryable_gemini_error(asyncio.TimeoutError()) is True
    assert _is_retryable_gemini_error(TimeoutError()) is True
    assert _is_retryable_gemini_error(httpx.ConnectTimeout("connection timed out")) is True
    assert _is_retryable_gemini_error(httpx.NetworkError("connection reset")) is True

    # 429 rate limit is retryable
    err_429 = errors.ClientError(429, {"message": "Resource exhausted"}, None)
    assert _is_retryable_gemini_error(err_429) is True

    # 5xx server errors are retryable
    err_503 = errors.ServerError(503, {"message": "Unavailable"}, None)
    assert _is_retryable_gemini_error(err_503) is True

    # 404 model not found is NOT retryable (fail immediately)
    err_404 = errors.ClientError(404, {"message": "Model not found"}, None)
    assert _is_retryable_gemini_error(err_404) is False

    # 401 auth error is NOT retryable
    err_401 = errors.ClientError(401, {"message": "API key invalid"}, None)
    assert _is_retryable_gemini_error(err_401) is False


@pytest.mark.asyncio
async def test_gemini_does_not_retry_model_not_found(gemini_settings: Settings) -> None:
    provider = GeminiLLMProvider(gemini_settings)
    err_404 = errors.ClientError(
        404, {"message": "models/gemini-2.5-flash is no longer available"}, None
    )

    mock_gen = AsyncMock(side_effect=err_404)
    with patch.object(provider._client.aio.models, "generate_content", new=mock_gen):
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(req)
        # Verify it was called only once (no useless retries on 404)
        assert mock_gen.call_count == 1
        assert "404" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_provider_timeout_504_raises_provider_timeout_error(
    gemini_settings: Settings,
) -> None:
    provider = GeminiLLMProvider(gemini_settings)
    err_504 = errors.ServerError(
        504, {"message": "Deadline expired before operation could complete."}, None
    )

    mock_gen = AsyncMock(side_effect=err_504)
    with patch.object(provider._client.aio.models, "generate_content", new=mock_gen):
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await provider.generate(req)
        assert "504" in str(exc_info.value)


def test_parse_gemini_retry_delay_from_rpc_and_human() -> None:
    from app.ai.providers.gemini_provider import _parse_gemini_retry_delay

    # RPC details format: retryDelay: '71416s'
    rpc_err = "429 RESOURCE_EXHAUSTED. {'details': [{'retryDelay': '71416s'}]}"
    assert _parse_gemini_retry_delay(rpc_err) == 71416.0

    # Human message format: 'Please retry in 19h50m16s'
    human_err = "You exceeded your current quota. Please retry in 19h50m16s."
    expected = 19 * 3600 + 50 * 60 + 16
    assert _parse_gemini_retry_delay(human_err) == expected

    # Short burst format: 'retry in 5s'
    assert _parse_gemini_retry_delay("Please retry in 5s") == 5.0

    # No delay present
    assert _parse_gemini_retry_delay("Generic rate limit without duration") is None


def test_is_not_retryable_when_quota_reset_is_long() -> None:
    long_quota_err = errors.ClientError(
        429,
        {
            "message": "Quota exceeded. Please retry in 19h50m16s.",
            "details": [{"retryDelay": "71416s"}],
        },
        None,
    )
    # Long delay (> 10s) must NOT be retried
    assert _is_retryable_gemini_error(long_quota_err) is False


@pytest.mark.asyncio
async def test_gemini_does_not_retry_long_quota_delay(gemini_settings: Settings) -> None:
    provider = GeminiLLMProvider(gemini_settings)
    err_429 = errors.ClientError(
        429,
        {
            "message": "Quota exceeded for metric: limit 20. Please retry in 19h50m16s.",
            "details": [{"retryDelay": "71416s"}],
        },
        None,
    )

    mock_gen = AsyncMock(side_effect=err_429)
    with patch.object(provider._client.aio.models, "generate_content", new=mock_gen):
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(req)
        # Verify it was called only once (no repeated retries when reset is ~20 hours away)
        assert mock_gen.call_count == 1
        assert "429" in str(exc_info.value)
        assert "quota reset in 71416s" in str(exc_info.value)


def test_gemini_model_configuration_default_3_8() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key=SecretStr("AIzaSyTestMockKey1234567890abcdef"),
        gemini_model="gemini-3.8-flash",
    )
    provider = GeminiLLMProvider(settings)
    assert provider._model == "gemini-3.8-flash"


@pytest.mark.asyncio
async def test_fallback_llm_provider_triggers_on_429_quota() -> None:
    from app.ai.providers.fallback_provider import FallbackLLMProvider
    from app.ai.providers.mock import MockLLMProvider

    primary = MagicMock()
    primary.provider_name = "gemini"
    primary.generate = AsyncMock(
        side_effect=ProviderError("Gemini quota/rate limit exceeded (HTTP 429): Quota exceeded")
    )

    fallback = MockLLMProvider()
    resilient = FallbackLLMProvider(primary=primary, fallback=fallback, is_production=False)

    req = GenerationRequest(
        messages=[
            Message(role="system", content="LEO context passages"),
            Message(role="user", content="Tell me about fees"),
        ]
    )
    res = await resilient.generate(req)
    assert res is not None
    assert primary.generate.call_count == 1


def test_fallback_llm_provider_rejects_mock_in_production() -> None:
    from app.ai.providers.fallback_provider import FallbackLLMProvider
    from app.ai.providers.mock import MockLLMProvider

    primary = MagicMock()
    primary.provider_name = "gemini"
    fallback = MockLLMProvider()

    with pytest.raises(
        ValueError, match="Mock LLM provider cannot be used as a fallback in production"
    ):
        FallbackLLMProvider(primary=primary, fallback=fallback, is_production=True)

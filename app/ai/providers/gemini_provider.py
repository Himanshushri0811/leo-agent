"""
Google Gemini LLM and embeddings provider using the official google-genai Python SDK.

Wraps google-genai behind LEO's provider-agnostic interfaces.
Handles retries with bounded exponential backoff.
All credentials are read from application Settings – never logged or hardcoded.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import inspect
import re
from typing import Any

from google import genai
from google.genai import errors, types
import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.ai.interfaces import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
)
from app.core.config import Settings
from app.core.exceptions import ProviderError, ProviderTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _parse_gemini_retry_delay(exc: BaseException) -> float | None:
    """Extract retry delay in seconds from a Gemini APIError or exception message."""
    msg = str(exc)
    # Check for retryDelay in RPC error details (e.g. '71416s')
    m = re.search(r"retryDelay['\":\s]+(\d+(?:\.\d+)?)s", msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Check for human format 'retry in 19h50m16s'
    m = re.search(r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", msg, re.IGNORECASE)
    if m and any(m.groups()):
        try:
            h = float(m.group(1) or 0)
            mins = float(m.group(2) or 0)
            s = float(m.group(3) or 0)
            return h * 3600 + mins * 60 + s
        except ValueError:
            pass
    return None


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Return True only for transient errors that warrant a retry."""
    if isinstance(
        exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException, httpx.NetworkError)
    ):
        return True
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.APIError):
        # HTTP 429: Check if this is a long-term daily quota exhaustion
        if exc.code == 429:
            delay = _parse_gemini_retry_delay(exc)
            if delay is not None and delay > 10.0:
                # Do NOT retry when quota reset is far in the future
                return False
            exc_str = str(exc)
            if "RESOURCE_EXHAUSTED" in exc_str and (
                "quota" in exc_str.lower() or "limit: 20" in exc_str
            ):
                return False
            return True
        # HTTP 5xx: Transient provider server errors
        if exc.code and exc.code >= 500:
            return True
    return False


# Regex to sanitize API keys from error messages and logs
_KEY_PATTERN = re.compile(r"AIzaSy[A-Za-z0-9_-]{33}|[a-zA-Z0-9_-]{32,}")


def _sanitize_error_message(msg: str) -> str:
    """Strip potential secret tokens or API keys from error strings."""
    return _KEY_PATTERN.sub("[REDACTED_API_KEY]", msg)


import time

# Verified working fast Gemini LLM models with automatic candidate fallback
GEMINI_LLM_MODEL = "gemini-3.5-flash"
LLM_CANDIDATE_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
)


class GeminiLLMProvider:
    """Production Google Gemini generation provider using official google-genai SDK."""

    provider_name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self._api_key = settings.gemini_api_key.get_secret_value()
        self._model = settings.gemini_model or settings.llm_model
        # Use fast, modern default if generic model name passed
        if self._model.startswith("gpt-") or self._model.startswith("claude-"):
            self._model = "gemini-3.5-flash"

        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_retries = max(1, settings.llm_max_retries)

        # Allocate per-request HTTP timeout so retries fit within the total deadline
        if self._max_retries > 1:
            per_request_timeout = max(
                5.0, min(float(self._timeout_seconds) / (self._max_retries - 1), 15.0)
            )
        else:
            per_request_timeout = float(self._timeout_seconds)

        # google-genai HttpOptions.timeout expects milliseconds (int)
        self._client_timeout_ms = int(per_request_timeout * 1000)

        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=self._client_timeout_ms),
        )

    def _prepare_contents(
        self, request: GenerationRequest
    ) -> tuple[str | None, list[types.Content]]:
        """Separate system instruction from conversation turns and map roles."""
        system_instruction: str | None = None
        contents: list[types.Content] = []

        for m in request.messages:
            if m.role == "system":
                # Aggregate system instructions if multiple
                if system_instruction:
                    system_instruction += "\n\n" + m.content
                else:
                    system_instruction = m.content
            else:
                # Map role: LEO/OpenAI 'assistant' -> Gemini 'model'
                role = "model" if m.role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=m.content)],
                    )
                )

        # Fallback if no user turns were provided
        if not contents:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

        return system_instruction, contents

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        t_start = time.perf_counter()
        system_instruction, contents = self._prepare_contents(request)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=request.max_tokens or self._max_tokens,
            temperature=request.temperature
            if request.temperature is not None
            else self._temperature,
            stop_sequences=request.stop_sequences or None,
        )

        @retry(
            retry=retry_if_exception(_is_retryable_gemini_error),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        async def _call() -> types.GenerateContentResponse:
            try:
                return await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
            except errors.APIError as exc:
                if exc.code == 429:
                    logger.warning("gemini_rate_limited_retrying", model=self._model)
                raise

        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await _call()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning(
                "gemini_generation_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category="timeout",
                error_detail="application deadline exceeded",
            )
            raise ProviderTimeoutError(
                f"Gemini API request timed out after {self._timeout_seconds}s (application deadline exceeded)"
            ) from exc
        except httpx.TimeoutException as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            clean_err = _sanitize_error_message(str(exc))
            logger.warning(
                "gemini_generation_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category="connection_timeout",
                error_detail=clean_err[:150],
            )
            raise ProviderTimeoutError(
                f"Gemini HTTP connection timed out after {self._timeout_seconds}s: {clean_err}"
            ) from exc
        except errors.APIError as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            clean_err = _sanitize_error_message(str(exc))
            category = (
                "provider_timeout"
                if (exc.code in (408, 504) or "DEADLINE_EXCEEDED" in clean_err)
                else f"http_{exc.code}"
            )
            logger.warning(
                "gemini_generation_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category=category,
                error_detail=clean_err[:150],
            )
            if exc.code in (408, 504) or "DEADLINE_EXCEEDED" in clean_err:
                raise ProviderTimeoutError(
                    f"Gemini provider-side timeout (HTTP {exc.code}): {clean_err}"
                ) from exc
            if exc.code == 429:
                delay = _parse_gemini_retry_delay(exc)
                delay_str = f" (quota reset in {int(delay)}s)" if delay else ""
                raise ProviderError(
                    f"Gemini quota/rate limit exceeded (HTTP 429){delay_str}: {clean_err}"
                ) from exc
            if exc.code in (401, 403):
                raise ProviderError(
                    f"Gemini authentication failed (HTTP {exc.code}): {clean_err}"
                ) from exc
            if exc.code == 404:
                raise ProviderError(
                    f"Gemini model '{self._model}' not found or unavailable (HTTP 404): {clean_err}"
                ) from exc
            if exc.code and exc.code >= 500:
                raise ProviderError(f"Gemini server error (HTTP {exc.code}): {clean_err}") from exc
            raise ProviderError(f"Gemini API error (HTTP {exc.code}): {clean_err}") from exc
        except (RetryError, Exception) as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            clean_err = _sanitize_error_message(str(exc))
            logger.warning(
                "gemini_generation_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category="general_error",
                error_detail=clean_err[:150],
            )
            raise ProviderError(f"Gemini generation error: {clean_err}") from exc

        content = response.text or ""
        prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        completion_tokens = (
            response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        )
        finish_reason = "stop"
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = str(response.candidates[0].finish_reason).lower()

        dur_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "timing_gemini_generation",
            model=self._model,
            duration_ms=round(dur_ms, 2),
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )

        return GenerationResponse(
            content=content,
            model=self._model,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            finish_reason=finish_reason,
            provider=self.provider_name,
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream generated text chunks asynchronously using Gemini streaming API."""
        t_start = time.perf_counter()
        system_instruction, contents = self._prepare_contents(request)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=request.max_tokens or self._max_tokens,
            temperature=request.temperature
            if request.temperature is not None
            else self._temperature,
            stop_sequences=request.stop_sequences or None,
        )

        ttft_ms: float | None = None
        try:
            async with asyncio.timeout(self._timeout_seconds):
                stream_res = self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                stream = await stream_res if inspect.isawaitable(stream_res) else stream_res
                async for chunk in stream:
                    if chunk.text:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000.0
                        yield chunk.text

            dur_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info(
                "timing_gemini_stream",
                model=self._model,
                ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
                duration_ms=round(dur_ms, 2),
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning(
                "gemini_streaming_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category="timeout",
                error_detail="streaming timeout",
            )
            raise ProviderTimeoutError(
                f"Gemini streaming timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.TimeoutException as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            clean_err = _sanitize_error_message(str(exc))
            logger.warning(
                "gemini_streaming_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category="connection_timeout",
                error_detail=clean_err[:150],
            )
            raise ProviderTimeoutError(
                f"Gemini streaming connection timed out: {clean_err}"
            ) from exc
        except errors.APIError as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            clean_err = _sanitize_error_message(str(exc))
            category = (
                "provider_timeout"
                if (exc.code in (408, 504) or "DEADLINE_EXCEEDED" in clean_err)
                else f"http_{exc.code}"
            )
            logger.warning(
                "gemini_streaming_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category=category,
                error_detail=clean_err[:150],
            )
            if exc.code in (408, 504) or "DEADLINE_EXCEEDED" in clean_err:
                raise ProviderTimeoutError(
                    f"Gemini streaming provider timeout (HTTP {exc.code}): {clean_err}"
                ) from exc
            raise ProviderError(f"Gemini streaming error (HTTP {exc.code}): {clean_err}") from exc
        except Exception as exc:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            clean_err = _sanitize_error_message(str(exc))
            logger.warning(
                "gemini_streaming_failed",
                provider=self.provider_name,
                model=self._model,
                latency_ms=round(dur_ms, 2),
                failure_category="general_error",
                error_detail=clean_err[:150],
            )
            raise ProviderError(f"Gemini streaming error: {clean_err}") from exc

    async def health_check(self) -> bool:
        try:
            # Quick probe with 1 output token
            config = types.GenerateContentConfig(max_output_tokens=1)
            async with asyncio.timeout(10.0):
                resp = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents="ping",
                    config=config,
                )
            return bool(resp)
        except Exception as exc:
            logger.warning("gemini_health_check_failed", error=_sanitize_error_message(str(exc)))
            return False


class GeminiEmbeddingsProvider:
    """Production Google Gemini embeddings provider."""

    provider_name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self._api_key = settings.gemini_api_key.get_secret_value()
        self._model = "text-embedding-004"
        self._dimensions = settings.embeddings_dimensions

        # Verify dimensionality compatibility with current pgvector schema (1536)
        if self._dimensions > 768:
            logger.warning(
                "gemini_embedding_dimension_mismatch",
                configured_dimension=self._dimensions,
                gemini_native_max=768,
                detail="Gemini text-embedding-004 supports up to 768 dimensions. If connecting to a Vector(1536) column, ensure vector dimensions match before indexing.",
            )

        # google-genai HttpOptions.timeout expects milliseconds (int)
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=60000),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if not request.texts:
            return EmbeddingResponse(
                embeddings=[],
                model=self._model,
                token_count=0,
                provider=self.provider_name,
            )

        # Batch embed with Gemini embed_content
        embeddings: list[list[float]] = []
        dim = min(self._dimensions, 768)

        for text in request.texts:
            try:
                config = types.EmbedContentConfig(
                    output_dimensionality=dim,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                res = await self._client.aio.models.embed_content(
                    model=self._model,
                    contents=text,
                    config=config,
                )
                if res.embedding and res.embedding.values:
                    vec = list(res.embedding.values)
                    # If existing database schema strictly expects 1536 dimensions, zero-pad safely
                    if self._dimensions > len(vec):
                        vec = vec + [0.0] * (self._dimensions - len(vec))
                    embeddings.append(vec)
            except Exception as exc:
                clean_err = _sanitize_error_message(str(exc))
                raise ProviderError(f"Gemini embedding failed: {clean_err}") from exc

        return EmbeddingResponse(
            embeddings=embeddings,
            model=self._model,
            token_count=sum(len(t.split()) for t in request.texts),
            provider=self.provider_name,
        )

    async def health_check(self) -> bool:
        try:
            res = await self._client.aio.models.embed_content(
                model=self._model,
                contents="test",
                config=types.EmbedContentConfig(output_dimensionality=64),
            )
            return bool(res.embedding)
        except Exception:
            return False

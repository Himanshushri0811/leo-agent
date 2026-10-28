"""
OpenAI LLM and embeddings provider.

Wraps the openai SDK behind the provider-agnostic interfaces.
Retries transient errors with bounded exponential backoff.
API key is read from settings - never hardcoded.
"""

from __future__ import annotations

import openai
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
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

_RETRYABLE = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)


class OpenAILLMProvider:
    """Production OpenAI generation provider."""

    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # We handle retries with tenacity
        )
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._max_retries = settings.llm_max_retries

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _call() -> openai.types.chat.ChatCompletion:
            return await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

        try:
            response = await _call()
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except (openai.OpenAIError, RetryError) as exc:
            raise ProviderError(str(exc)) from exc

        choice = response.choices[0]
        return GenerationResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
            provider=self.provider_name,
        )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False


class OpenAIEmbeddingsProvider:
    """Production OpenAI embeddings provider."""

    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=60,
            max_retries=0,
        )
        self._model = settings.embeddings_model
        self._dimensions = settings.embeddings_dimensions

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if not request.texts:
            return EmbeddingResponse(
                embeddings=[], model=self._model, token_count=0, provider=self.provider_name
            )
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=request.texts,
                dimensions=self._dimensions,
            )
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except openai.OpenAIError as exc:
            raise ProviderError(str(exc)) from exc

        embeddings = [d.embedding for d in response.data]
        total_tokens = response.usage.total_tokens if response.usage else 0
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self._model,
            token_count=total_tokens,
            provider=self.provider_name,
        )

    async def health_check(self) -> bool:
        try:
            await self._client.models.retrieve(self._model)
            return True
        except Exception:
            return False

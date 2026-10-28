"""
Anthropic Claude LLM provider.

Maps the provider-agnostic GenerationRequest to the Anthropic Messages API.
System messages are extracted and passed as the top-level `system` parameter.
"""

from __future__ import annotations

import anthropic
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.ai.interfaces import GenerationRequest, GenerationResponse
from app.core.config import Settings
from app.core.exceptions import ProviderError, ProviderTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


class AnthropicLLMProvider:
    """Production Anthropic Claude generation provider."""

    provider_name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        self._max_retries = settings.llm_max_retries

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        system_parts = [m.content for m in request.messages if m.role == "system"]
        system_prompt = "\n\n".join(system_parts) if system_parts else anthropic.NOT_GIVEN
        messages = [
            {"role": m.role, "content": m.content} for m in request.messages if m.role != "system"
        ]

        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _call() -> anthropic.types.Message:
            return await self._client.messages.create(
                model=self._model,
                max_tokens=request.max_tokens,
                system=system_prompt,
                messages=messages,  # type: ignore[arg-type]
            )

        try:
            response = await _call()
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except (anthropic.AnthropicError, RetryError) as exc:
            raise ProviderError(str(exc)) from exc

        content_text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return GenerationResponse(
            content=content_text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason or "stop",
            provider=self.provider_name,
        )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

"""
Fallback LLM Provider wrapper.

Wraps a primary and optional fallback LLM provider.
Only triggers fallback on upstream quota exhaustion (HTTP 429) or transient
availability/timeout failures.
Never falls back to mock responses in production.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.ai.interfaces import GenerationRequest, GenerationResponse, LLMProvider
from app.core.exceptions import ProviderError, ProviderTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)

_FALLBACK_ELIGIBLE_INDICATORS = (
    "429",
    "quota",
    "rate limit",
    "resource_exhausted",
    "502",
    "503",
    "504",
    "unavailable",
    "server error",
    "404",
    "not found",
    "timed out",
    "deadline exceeded",
)


class FallbackLLMProvider:
    """Resilient provider that falls back to a secondary provider upon quota or availability errors."""

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None = None,
        is_production: bool = False,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._is_production = is_production

        if self._is_production and self._fallback and getattr(self._fallback, "provider_name", "") == "mock":
            raise ValueError("Mock LLM provider cannot be used as a fallback in production.")

    @property
    def provider_name(self) -> str:
        return self._primary.provider_name

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def fallback(self) -> LLMProvider | None:
        return self._fallback

    def _is_eligible_for_fallback(self, exc: Exception) -> bool:
        if isinstance(exc, ProviderTimeoutError):
            return True
        msg = str(exc).lower()
        return any(ind in msg for ind in _FALLBACK_ELIGIBLE_INDICATORS)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            return await self._primary.generate(request)
        except Exception as exc:
            if self._fallback and self._is_eligible_for_fallback(exc):
                logger.warning(
                    "llm_primary_failed_triggering_fallback",
                    primary=self._primary.provider_name,
                    fallback=self._fallback.provider_name,
                    error=str(exc),
                )
                return await self._fallback.generate(request)
            raise

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        try:
            async for chunk in self._primary.generate_stream(request):
                yield chunk
        except Exception as exc:
            if self._fallback and self._is_eligible_for_fallback(exc):
                logger.warning(
                    "llm_primary_stream_failed_triggering_fallback",
                    primary=self._primary.provider_name,
                    fallback=self._fallback.provider_name,
                    error=str(exc),
                )
                async for chunk in self._fallback.generate_stream(request):
                    yield chunk
            else:
                raise

    async def health_check(self) -> bool:
        return await self._primary.health_check()

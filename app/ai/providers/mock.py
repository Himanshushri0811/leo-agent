"""
Mock LLM and embeddings providers for testing and local development.

Returns deterministic, safe responses without any network calls.
Tests must use these providers; real providers require opt-in markers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import hashlib

from app.ai.interfaces import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
)


class MockLLMProvider:
    """Deterministic mock LLM. Does not call any external API."""

    provider_name = "mock"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        system_msg = next((m.content for m in request.messages if m.role == "system"), "")
        passages_text = ""
        if "--- BEGIN KNOWLEDGE PASSAGES ---" in system_msg:
            try:
                passages_text = system_msg.split("--- BEGIN KNOWLEDGE PASSAGES ---")[1].split(
                    "--- END KNOWLEDGE PASSAGES ---"
                )[0].strip()
            except IndexError:
                passages_text = ""

        if passages_text and passages_text != "(none)":
            content = (
                f"[Mock LEO response] Bitmax Group: Based on approved company knowledge, "
                f"{passages_text[:250]}... "
                "For support, contact inquiries@bitmaxgroup.com or visit https://support.bitmaxgroup.com."
            )
        else:
            content = (
                "[Mock LEO response] I don't currently have detailed information on that topic in "
                "Bitmax Group's approved knowledge base. Please contact our team directly at "
                "inquiries@bitmaxgroup.com or visit https://support.bitmaxgroup.com."
            )

        return GenerationResponse(
            content=content,
            model="mock-model",
            prompt_tokens=len(last_user) // 4,
            completion_tokens=len(content) // 4,
            finish_reason="stop",
            provider=self.provider_name,
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        response = await self.generate(request)
        words = response.content.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    async def health_check(self) -> bool:
        return True


class MockEmbeddingsProvider:
    """Deterministic mock embeddings. Produces reproducible vectors with lexical similarity."""

    provider_name = "mock"
    DIMENSIONS = 1536

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        embeddings = [self._text_to_vector(t) for t in request.texts]
        return EmbeddingResponse(
            embeddings=embeddings,
            model="mock-embedding-model",
            token_count=sum(len(t.split()) for t in request.texts),
            provider=self.provider_name,
        )

    async def health_check(self) -> bool:
        return True

    def _text_to_vector(self, text: str) -> list[float]:
        """Produce a deterministic, reproducible vector reflecting lexical similarity in mock mode."""
        import math
        import re

        tokens = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        vec = [0.0] * self.DIMENSIONS

        # Token semantic activation
        for token in sorted(tokens):
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            for i in range(32):
                idx = (h + i * 47) % self.DIMENSIONS
                weight = 1.0 if ((h >> (i % 64)) & 1) else -1.0
                vec[idx] += weight

        # Global sentence digest base
        digest = hashlib.sha256(text.encode()).digest()
        raw = (digest * (self.DIMENSIONS // len(digest) + 1))[: self.DIMENSIONS]
        for i, b in enumerate(raw):
            vec[i] += (b - 127.5) / 127.5 * 0.2

        norm = math.sqrt(sum(v * v for v in vec))
        if norm < 1e-9:
            return [1.0] + [0.0] * (self.DIMENSIONS - 1)
        return [v / norm for v in vec]

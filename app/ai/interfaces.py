"""
Provider-agnostic interfaces for AI capabilities.

All concrete providers must implement these protocols.
The rest of the application only imports from this module,
never from a specific provider module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Message:
    """A single conversation turn."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class GenerationRequest:
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.2
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length" | "error"
    provider: str


@dataclass
class EmbeddingRequest:
    texts: list[str]


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    model: str
    token_count: int
    provider: str


@dataclass
class AudioTranscriptionRequest:
    """Speech-to-text request container."""

    audio_data: bytes
    audio_format: str = "wav"  # "wav" | "mp3" | "pcm" | "webm"
    language_hint: str = "en-IN"  # e.g., "en-IN" for Indian English, "hi-IN" for Hindi


@dataclass
class AudioTranscriptionResponse:
    """Speech-to-text response container."""

    text: str
    confidence: float
    detected_language: str
    provider: str


@dataclass
class SpeechSynthesisRequest:
    """Text-to-speech synthesis request."""

    text: str
    voice_name: str = "Aoede"  # Warm, composed persona voice
    language_code: str = "en-IN"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    audio_format: str = "audio/wav"  # "audio/wav" | "audio/mp3" | "audio/pcm"


@dataclass
class SpeechSynthesisResponse:
    """Text-to-speech synthesis response."""

    audio_data: bytes
    audio_format: str
    duration_seconds: float
    provider: str
    sample_rate: int = 24000


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for language model generation providers."""

    @property
    def provider_name(self) -> str: ...

    async def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream generated text chunks asynchronously."""
        ...

    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
        ...


@runtime_checkable
class EmbeddingsProvider(Protocol):
    """Protocol for text embedding providers."""

    @property
    def provider_name(self) -> str: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

    async def health_check(self) -> bool: ...


@runtime_checkable
class STTProvider(Protocol):
    """Protocol for speech-to-text providers."""

    @property
    def provider_name(self) -> str: ...

    async def transcribe(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse: ...

    async def health_check(self) -> bool: ...


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for text-to-speech providers."""

    @property
    def provider_name(self) -> str: ...

    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResponse: ...

    async def synthesize_stream(
        self, request: SpeechSynthesisRequest
    ) -> AsyncIterator[bytes]: ...

    async def health_check(self) -> bool: ...

"""
Deterministic Mock TTS and STT providers for testing and local development.

Produces valid, playable PCM WAV audio and simulated transcripts without network calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import math
import struct

from app.ai.interfaces import (
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    SpeechSynthesisRequest,
    SpeechSynthesisResponse,
    STTProvider,
    TTSProvider,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def create_mock_wav(
    duration_seconds: float = 1.0,
    sample_rate: int = 24000,
    frequency: float = 440.0,
) -> bytes:
    """Generate a standard 44-byte RIFF/WAVE header and 16-bit mono PCM sine wave audio."""
    num_samples = max(int(sample_rate * duration_seconds), 240)
    byte_rate = sample_rate * 2  # 16-bit mono = 2 bytes per sample
    block_align = 2
    data_size = num_samples * 2
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM format subchunk size
        1,  # AudioFormat 1 = PCM
        1,  # NumChannels 1 = Mono
        sample_rate,
        byte_rate,
        block_align,
        16,  # BitsPerSample = 16
        b"data",
        data_size,
    )

    # Low-volume sine wave (0.1 amplitude)
    samples: list[bytes] = []
    for i in range(num_samples):
        sample_val = int(32767.0 * 0.08 * math.sin(2.0 * math.pi * frequency * (i / sample_rate)))
        samples.append(struct.pack("<h", sample_val))

    return header + b"".join(samples)


class MockTTSProvider:
    """Deterministic mock Text-to-Speech provider generating valid synthetic WAV audio."""

    provider_name = "mock"

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResponse:
        words = len(request.text.split())
        # Average reading rate: ~2.5 words per second
        duration = max(0.4, min(30.0, words / 2.5))
        audio_data = create_mock_wav(
            duration_seconds=duration,
            sample_rate=self.sample_rate,
            frequency=520.0 if "Aoede" in request.voice_name else 440.0,
        )
        return SpeechSynthesisResponse(
            audio_data=audio_data,
            audio_format="audio/wav",
            duration_seconds=round(duration, 2),
            provider=self.provider_name,
            sample_rate=self.sample_rate,
        )

    async def synthesize_stream(
        self, request: SpeechSynthesisRequest
    ) -> AsyncIterator[bytes]:
        full_res = await self.synthesize(request)
        chunk_size = 2048
        data = full_res.audio_data
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    async def health_check(self) -> bool:
        return True


class MockSTTProvider:
    """Deterministic mock Speech-to-Text provider."""

    provider_name = "mock"

    async def transcribe(
        self, request: AudioTranscriptionRequest
    ) -> AudioTranscriptionResponse:
        # Check audio length for basic validity
        if len(request.audio_data) < 44:
            return AudioTranscriptionResponse(
                text="",
                confidence=0.0,
                detected_language=request.language_hint or "en-IN",
                provider=self.provider_name,
            )

        # Default simulated customer utterance in Indian English
        sample_text = "Hello, can you tell me about Bitmax Group's cloud and AI services?"
        return AudioTranscriptionResponse(
            text=sample_text,
            confidence=0.98,
            detected_language=request.language_hint or "en-IN",
            provider=self.provider_name,
        )

    async def health_check(self) -> bool:
        return True

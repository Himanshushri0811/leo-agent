"""Unit tests for the TTS audio generation pipeline.

Tests cover:
1. L16 PCM -> WAV header wrapping (_l16_to_wav)
2. Sample rate parsing from Gemini MIME types (_parse_l16_sample_rate)
3. WAV wrapping detection (_needs_wav_wrapping)
4. GeminiTTSProvider correct model default and ms timeout
5. GeminiTTSProvider wraps L16 response in WAV
6. MockTTSProvider produces valid WAV for synthesize_turn_audio
7. DemoCallService.synthesize_turn_audio integration & error handling
"""
from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.voice.gemini_voice import (
    GEMINI_TTS_MODEL,
    GeminiTTSProvider,
    _l16_to_wav,
    _needs_wav_wrapping,
    _parse_l16_sample_rate,
)
from app.voice.mock_voice import MockTTSProvider
from app.ai.interfaces import SpeechSynthesisRequest
from app.core.config import AppEnv, Settings, TTSProvider
from app.services.demo_call_service import DemoCallService


def test_l16_to_wav_produces_valid_riff_header() -> None:
    raw_pcm = b"\x00\x01" * 100
    wav = _l16_to_wav(raw_pcm, sample_rate=24000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    audio_fmt = struct.unpack("<H", wav[20:22])[0]
    assert audio_fmt == 1  # PCM
    channels = struct.unpack("<H", wav[22:24])[0]
    assert channels == 1  # Mono
    sample_rate = struct.unpack("<I", wav[24:28])[0]
    assert sample_rate == 24000
    bits = struct.unpack("<H", wav[34:36])[0]
    assert bits == 16
    assert b"data" in wav
    assert len(wav) == 44 + len(raw_pcm)


def test_parse_l16_sample_rate() -> None:
    assert _parse_l16_sample_rate("audio/L16;rate=24000") == 24000
    assert _parse_l16_sample_rate("audio/l16;rate=16000") == 16000
    assert _parse_l16_sample_rate("audio/L16") == 24000
    assert _parse_l16_sample_rate("audio/wav") == 24000
    assert _parse_l16_sample_rate("") == 24000


def test_needs_wav_wrapping() -> None:
    # WAV mime does not need wrapping
    assert not _needs_wav_wrapping("audio/wav")
    assert not _needs_wav_wrapping("audio/x-wav")

    # L16 mime needs wrapping
    assert _needs_wav_wrapping("audio/L16;rate=24000")
    assert _needs_wav_wrapping("audio/l16")
    assert _needs_wav_wrapping("audio/pcm")


def test_gemini_tts_provider_uses_flash_model() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.GEMINI,
    )
    provider = GeminiTTSProvider(settings)
    assert provider._model == GEMINI_TTS_MODEL
    assert GEMINI_TTS_MODEL == "gemini-3.8-flash-lite-tts"


def test_gemini_tts_timeout_converted_to_ms() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        llm_timeout_seconds=12,
    )
    provider = GeminiTTSProvider(settings)
    assert provider._timeout_seconds == 12.0
    assert provider._client_timeout_ms == 12000.0


@pytest.mark.asyncio
async def test_mock_tts_provider_synthesizes_valid_wav() -> None:
    provider = MockTTSProvider()
    response = await provider.synthesize(SpeechSynthesisRequest(text="Hello from LEO"))
    assert response.audio_data[:4] == b"RIFF"
    assert response.audio_format == "audio/wav"
    assert response.duration_seconds > 0
    assert len(response.audio_data) > 44


@pytest.mark.asyncio
async def test_gemini_tts_wraps_raw_l16_in_wav() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.GEMINI,
    )
    provider = GeminiTTSProvider(settings)

    # Mock the client response returning raw L16 PCM
    raw_pcm = b"\x01\x00" * 4800  # 0.2s of audio at 24kHz 16-bit mono
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(
        mime_type="audio/L16;rate=24000",
        data=raw_pcm,
    )
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=mock_response),
    ):
        result = await provider.synthesize(
            SpeechSynthesisRequest(text="Thank you for calling.")
        )
        assert result.audio_data[:4] == b"RIFF"
        assert result.audio_format == "audio/wav"
        assert len(result.audio_data) == 44 + len(raw_pcm)
        assert result.sample_rate == 24000


@pytest.mark.asyncio
async def test_demo_call_service_synthesize_turn_audio() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.MOCK,
    )
    service = DemoCallService(settings=settings)
    wav_bytes = await service.synthesize_turn_audio("Hello, thank you for calling LEO support.")
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert len(wav_bytes) > 44


@pytest.mark.asyncio
async def test_demo_call_service_synthesize_turn_audio_no_tts_raises() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.ELEVENLABS,
    )
    service = DemoCallService(settings=settings)
    with pytest.raises(RuntimeError, match="TTS provider is not configured"):
        await service.synthesize_turn_audio("Test message")

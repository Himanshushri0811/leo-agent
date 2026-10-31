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
        result = await provider.synthesize(SpeechSynthesisRequest(text="Thank you for calling."))
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


@pytest.mark.asyncio
async def test_gemini_tts_synthesize_passes_only_dialogue_without_preamble() -> None:
    """Verify synthesize() sends only request.text as contents and instructions via system_instruction."""
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.GEMINI,
    )
    provider = GeminiTTSProvider(settings)

    captured_call = {}

    raw_pcm = b"\x01\x00" * 2400
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(mime_type="audio/L16;rate=24000", data=raw_pcm)
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    async def mock_generate_content(*args, **kwargs):
        captured_call["model"] = kwargs.get("model")
        captured_call["contents"] = kwargs.get("contents")
        captured_call["config"] = kwargs.get("config")
        return mock_response

    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new=mock_generate_content,
    ):
        dialogue_text = "Bitmax Group provides cloud infrastructure and DevOps consulting."
        req = SpeechSynthesisRequest(
            text=dialogue_text,
            language_code="en-IN",
            voice_name="Aoede",
        )
        await provider.synthesize(req)

        # 1. contents must be ONLY the exact dialogue text, NEVER containing prompt leakage
        assert captured_call["contents"] == dialogue_text
        assert "Generate speech" not in captured_call["contents"]
        assert "Language:" not in captured_call["contents"]
        assert "The dialogue is:" not in captured_call["contents"]

        # 2. config must NOT use system_instruction (Gemini TTS models reject developer instructions with HTTP 400)
        cfg = captured_call["config"]
        assert cfg is not None
        assert getattr(cfg, "system_instruction", None) is None


@pytest.mark.asyncio
async def test_gemini_tts_synthesize_stream_passes_only_dialogue() -> None:
    """Verify synthesize_stream() passes only request.text and separates instructions."""
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.GEMINI,
    )
    provider = GeminiTTSProvider(settings)

    captured_call = {}

    async def mock_stream(*args, **kwargs):
        captured_call["contents"] = kwargs.get("contents")
        captured_call["config"] = kwargs.get("config")
        chunk = MagicMock()
        chunk.candidates = []
        yield chunk

    with patch.object(
        provider._client.aio.models,
        "generate_content_stream",
        new=mock_stream,
    ):
        dialogue_text = "Haan bilkul, main aapko services ke baare mein bata sakta hoon."
        req = SpeechSynthesisRequest(
            text=dialogue_text,
            language_code="hi-IN",
            voice_name="Aoede",
        )
        async for _ in provider.synthesize_stream(req):
            pass

        assert captured_call["contents"] == dialogue_text
        assert "Generate speech" not in captured_call["contents"]
        assert "The dialogue is:" not in captured_call["contents"]
        assert getattr(captured_call["config"], "system_instruction", None) is None


@pytest.mark.asyncio
async def test_gemini_tts_synthesize_stream_framed_passes_only_dialogue() -> None:
    """Verify synthesize_stream_framed() passes only request.text without preamble."""
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.GEMINI,
    )
    provider = GeminiTTSProvider(settings)

    captured_call = {}

    async def mock_stream(*args, **kwargs):
        captured_call["contents"] = kwargs.get("contents")
        captured_call["config"] = kwargs.get("config")
        chunk = MagicMock()
        part = MagicMock()
        part.inline_data = MagicMock(mime_type="audio/L16;rate=24000", data=b"\x00\x01" * 1200)
        chunk.candidates = [MagicMock(content=MagicMock(parts=[part]))]
        yield chunk

    with patch.object(
        provider._client.aio.models,
        "generate_content_stream",
        new=mock_stream,
    ):
        dialogue_text = "Our enterprise engineering team builds custom cloud platforms."
        req = SpeechSynthesisRequest(
            text=dialogue_text,
            language_code="en-IN",
            voice_name="Aoede",
        )
        frames = [f async for f in provider.synthesize_stream_framed(req)]
        assert len(frames) >= 2  # header + PCM + done
        assert captured_call["contents"] == dialogue_text
        assert "Generate speech" not in captured_call["contents"]
        assert "The dialogue" not in captured_call["contents"]


@pytest.mark.asyncio
async def test_gemini_tts_synthesize_empty_string_returns_silent_wav_immediately() -> None:
    """Verify empty text does not hit the model and returns a valid silent WAV."""
    settings = Settings(
        app_env=AppEnv.TESTING,
        gemini_api_key="test-key",
        tts_provider=TTSProvider.GEMINI,
    )
    provider = GeminiTTSProvider(settings)

    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new=AsyncMock(side_effect=AssertionError("Should not call API on empty text")),
    ):
        result = await provider.synthesize(SpeechSynthesisRequest(text="   "))
        assert result.audio_data[:4] == b"RIFF"
        assert result.duration_seconds == 0.0

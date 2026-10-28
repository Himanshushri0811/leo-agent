"""Unit tests for Voice architecture, TTS, STT, and turn-taking / barge-in."""

import pytest

from app.ai.interfaces import (
    AudioTranscriptionRequest,
    SpeechSynthesisRequest,
)
from app.core.config import AppEnv, Settings, STTProvider, TTSProvider
from app.voice import get_stt_provider, get_tts_provider
from app.voice.mock_voice import MockSTTProvider, MockTTSProvider, create_mock_wav
from app.voice.turn_manager import TurnManager, TurnState, UtteranceCancellationToken


def test_create_mock_wav_valid_header() -> None:
    wav_bytes = create_mock_wav(duration_seconds=0.5, sample_rate=24000)
    assert len(wav_bytes) > 44
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert wav_bytes[12:16] == b"fmt "
    assert b"data" in wav_bytes


@pytest.mark.asyncio
async def test_mock_tts_provider_synthesizes_playable_audio() -> None:
    tts = MockTTSProvider(sample_rate=24000)
    req = SpeechSynthesisRequest(
        text="Welcome to Bitmax Group. How can I help you today?",
        voice_name="Aoede",
        language_code="en-IN",
    )
    res = await tts.synthesize(req)
    assert res.provider == "mock"
    assert res.audio_format == "audio/wav"
    assert res.sample_rate == 24000
    assert res.duration_seconds > 0.0
    assert len(res.audio_data) > 44
    assert res.audio_data[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_mock_tts_provider_streaming() -> None:
    tts = MockTTSProvider()
    req = SpeechSynthesisRequest(text="Streaming test")
    stream = tts.synthesize_stream(req)
    chunks = [c async for c in stream]
    assert len(chunks) >= 1
    total_bytes = b"".join(chunks)
    assert total_bytes[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_mock_stt_provider_transcription() -> None:
    stt = MockSTTProvider()
    sample_audio = create_mock_wav(duration_seconds=0.5)
    req = AudioTranscriptionRequest(audio_data=sample_audio, language_hint="en-IN")
    res = await stt.transcribe(req)
    assert res.provider == "mock"
    assert res.confidence > 0.90
    assert "Bitmax Group" in res.text
    assert res.detected_language == "en-IN"


@pytest.mark.asyncio
async def test_turn_manager_barge_in_and_stream_cancellation() -> None:
    turn_mgr = TurnManager()
    session_id = "test_call_turn_01"

    # 1. Agent starts speaking turn
    agent_token = await turn_mgr.create_agent_turn(session_id)
    assert await turn_mgr.get_state(session_id) == TurnState.SPEAKING
    assert not agent_token.is_cancelled

    # 2. Simulated chunks generator
    async def mock_audio_stream():
        yield b"chunk_1"
        yield b"chunk_2"
        # Simulate user interruption happening right here
        await turn_mgr.on_user_speech_start(session_id)
        yield b"chunk_3_stale"
        yield b"chunk_4_stale"

    # 3. Filter speech stream with cancellation token
    filtered_stream = turn_mgr.filter_speech_stream(mock_audio_stream(), agent_token)
    yielded_chunks = [c async for c in filtered_stream]

    # Stale chunks after barge-in must be discarded
    assert agent_token.is_cancelled is True
    assert agent_token.cancelled_reason == "user_barge_in"
    assert b"chunk_1" in yielded_chunks
    assert b"chunk_3_stale" not in yielded_chunks
    assert b"chunk_4_stale" not in yielded_chunks
    assert await turn_mgr.get_state(session_id) == TurnState.LISTENING


def test_voice_factory_provider_selection() -> None:
    settings = Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        tts_provider=TTSProvider.MOCK,
        stt_provider=STTProvider.MOCK,
    )
    tts = get_tts_provider(settings)
    stt = get_stt_provider(settings)

    assert isinstance(tts, MockTTSProvider)
    assert isinstance(stt, MockSTTProvider)

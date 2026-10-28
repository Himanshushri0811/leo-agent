"""Voice provider factory and interfaces for LEO AI."""

from __future__ import annotations

from app.ai.interfaces import STTProvider, TTSProvider
from app.core.config import Settings, STTProvider as STTProviderEnum, TTSProvider as TTSProviderEnum


def get_tts_provider(settings: Settings) -> TTSProvider:
    """Return configured Text-to-Speech provider."""
    match settings.tts_provider:
        case TTSProviderEnum.GEMINI:
            from app.voice.gemini_voice import GeminiTTSProvider

            return GeminiTTSProvider(settings)
        case TTSProviderEnum.MOCK:
            from app.voice.mock_voice import MockTTSProvider

            return MockTTSProvider()
        case _:
            raise ValueError(f"Unknown TTS provider: {settings.tts_provider}")


def get_stt_provider(settings: Settings) -> STTProvider:
    """Return configured Speech-to-Text provider."""
    match settings.stt_provider:
        case STTProviderEnum.GEMINI:
            from app.voice.gemini_voice import GeminiSTTProvider

            return GeminiSTTProvider(settings)
        case STTProviderEnum.MOCK:
            from app.voice.mock_voice import MockSTTProvider

            return MockSTTProvider()
        case _:
            raise ValueError(f"Unknown STT provider: {settings.stt_provider}")

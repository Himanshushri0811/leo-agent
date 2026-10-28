"""
Streaming audio protocol for LEO voice pipeline.

Provides:
- Framed binary protocol for streaming raw PCM audio to the browser.
- Sentence/phrase boundary accumulation from LLM token streams.
- Per-turn timing instrumentation with correlation IDs.
- Cancellation token integration for stale audio rejection.
"""

from __future__ import annotations

import struct
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


# Frame type constants (1 byte, prefixed to each frame)
FRAME_TYPE_HEADER = 0x01
FRAME_TYPE_PCM = 0x02
FRAME_TYPE_TEXT = 0x03
FRAME_TYPE_DONE = 0x04
FRAME_TYPE_ERROR = 0x05


def pack_frame(frame_type: int, payload: bytes) -> bytes:
    """Pack a frame: [4-byte big-endian length][1-byte type][payload]."""
    return struct.pack(">I", 1 + len(payload)) + bytes([frame_type]) + payload


def pack_header(sample_rate: int, channels: int) -> bytes:
    """Pack the initial audio header frame (sample_rate + channels as uint32 each)."""
    payload = struct.pack(">II", sample_rate, channels)
    return pack_frame(FRAME_TYPE_HEADER, payload)


def pack_pcm_chunk(pcm_data: bytes) -> bytes:
    """Pack a raw PCM chunk frame."""
    return pack_frame(FRAME_TYPE_PCM, pcm_data)


def pack_text_chunk(text: str) -> bytes:
    """Pack a text transcript chunk frame."""
    return pack_frame(FRAME_TYPE_TEXT, text.encode("utf-8"))


def pack_done() -> bytes:
    """Pack a done frame."""
    return pack_frame(FRAME_TYPE_DONE, b"")


def pack_error(message: str) -> bytes:
    """Pack an error frame."""
    return pack_frame(FRAME_TYPE_ERROR, message.encode("utf-8"))


@dataclass
class TurnTiming:
    """Per-turn timing metrics with correlation ID for end-to-end latency tracing."""

    turn_id: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # LLM phase
    llm_start_ms: float | None = None
    llm_first_token_ms: float | None = None
    llm_complete_ms: float | None = None
    # TTS phase
    tts_first_sentence_start_ms: float | None = None
    tts_first_audio_byte_ms: float | None = None
    tts_complete_ms: float | None = None
    # Delivery phase
    audio_delivery_start_ms: float | None = None
    # Browser phase (filled by frontend)
    browser_playback_start_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "correlation_id": self.correlation_id,
            "llm_start_ms": round(self.llm_start_ms, 2) if self.llm_start_ms else None,
            "llm_first_token_ms": round(self.llm_first_token_ms, 2) if self.llm_first_token_ms else None,
            "llm_complete_ms": round(self.llm_complete_ms, 2) if self.llm_complete_ms else None,
            "tts_first_sentence_start_ms": round(self.tts_first_sentence_start_ms, 2) if self.tts_first_sentence_start_ms else None,
            "tts_first_audio_byte_ms": round(self.tts_first_audio_byte_ms, 2) if self.tts_first_audio_byte_ms else None,
            "tts_complete_ms": round(self.tts_complete_ms, 2) if self.tts_complete_ms else None,
            "audio_delivery_start_ms": round(self.audio_delivery_start_ms, 2) if self.audio_delivery_start_ms else None,
            "browser_playback_start_ms": round(self.browser_playback_start_ms, 2) if self.browser_playback_start_ms else None,
        }


class SentenceAccumulator:
    """Accumulates LLM token chunks and splits them into speakable sentences/phrases.

    Splits on sentence-ending punctuation (.!?) followed by whitespace,
    and also enforces a maximum phrase length to avoid overly long TTS inputs.
    """

    def __init__(self, max_phrase_chars: int = 120) -> None:
        self._buffer = ""
        self._max_phrase_chars = max_phrase_chars
        self._sentence_endings = (".", "!", "?")

    def add(self, token: str) -> list[str]:
        """Add a token and return any completed sentences/phrases."""
        self._buffer += token
        completed: list[str] = []

        while True:
            sentence = self._extract_sentence()
            if sentence is None:
                break
            completed.append(sentence)

        # Split overly long phrases at natural break points
        if len(self._buffer) > self._max_phrase_chars:
            split_at = self._find_split_point()
            if split_at > 0:
                completed.append(self._buffer[:split_at].strip())
                self._buffer = self._buffer[split_at:]

        return completed

    def _extract_sentence(self) -> str | None:
        """Extract a complete sentence from the buffer if one exists."""
        if not self._buffer.strip():
            return None

        for i, char in enumerate(self._buffer):
            if char in self._sentence_endings:
                if i + 1 >= len(self._buffer):
                    # Punctuation at end of buffer — wait for more tokens
                    return None
                next_char = self._buffer[i + 1]
                if next_char in (" ", "\n", "\t", '"', "'"):
                    sentence = self._buffer[: i + 1].strip()
                    self._buffer = self._buffer[i + 1 :]
                    return sentence

        return None

    def _find_split_point(self) -> int:
        """Find a good split point for long phrases (prefer comma, semicolon, space)."""
        search_start = max(0, self._max_phrase_chars - 40)
        search_end = min(len(self._buffer), self._max_phrase_chars + 40)
        region = self._buffer[search_start:search_end]

        for sep in (";", ",", " "):
            idx = region.rfind(sep)
            if idx >= 0:
                return search_start + idx + len(sep)

        return 0

    def flush(self) -> str | None:
        """Return any remaining text in the buffer."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None

    @property
    def pending(self) -> str:
        return self._buffer


async def stream_with_cancellation(
    source: AsyncIterator[Any],
    token: Any,
) -> AsyncIterator[Any]:
    """Yield items from source, stopping immediately if token is cancelled."""
    async for item in source:
        if getattr(token, "is_cancelled", False):
            logger.info(
                "stream_cancelled_mid_iteration",
                token_id=getattr(token, "token_id", "?"),
                reason=getattr(token, "cancelled_reason", "unknown"),
            )
            break
        yield item


def now_ms() -> float:
    """Current monotonic time in milliseconds."""
    return time.perf_counter() * 1000.0
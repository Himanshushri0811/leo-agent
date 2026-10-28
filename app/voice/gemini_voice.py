"""
Google Gemini Voice Providers: Text-to-Speech (TTS) and Speech-to-Text (STT).

Uses the official google-genai SDK for audio synthesis and transcription.
Supports configurable voices, bilingual English/Hindi support, and streaming.
All secrets are safely loaded from application Settings and sanitized on error.

TTS notes (verified against SDK v1.67.0, 2026-10-21):
  - Working TTS model: gemini-2.5-flash-preview-tts
  - The API returns raw L16 PCM (16-bit signed little-endian) with MIME type
    "audio/L16;codec=pcm;rate=24000". This module wraps it in a RIFF/WAVE
    header so browsers can play it without further transformation.
  - HttpOptions.timeout is in MILLISECONDS (SDK divides by 1000 internally).
"""

from __future__ import annotations

import asyncio
import inspect
import struct
from collections.abc import AsyncIterator
import re

from google import genai
from google.genai import errors, types

from app.ai.interfaces import (
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    SpeechSynthesisRequest,
    SpeechSynthesisResponse,
    STTProvider,
    TTSProvider,
)
from app.core.config import Settings
from app.core.exceptions import ProviderError, ProviderTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)

import time

# Verified working TTS models with fallback (fastest first for lowest latency)
GEMINI_TTS_MODEL = "gemini-3.8-flash-lite-tts"
TTS_CANDIDATE_MODELS = (
    "gemini-3.8-flash-lite-tts",
    "gemini-3.8-flash-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
)
GEMINI_STT_MODEL = "gemini-2.0-flash"

# Prebuilt voice options supported by Gemini TTS
SUPPORTED_GEMINI_VOICES = {
    "Aoede": "Warm, composed, clear customer-support voice (Recommended for LEO)",
    "Charon": "Deep, confident, calm baritone voice",
    "Puck": "Energetic, dynamic voice",
    "Fenrir": "Strong, authoritative voice",
    "Kore": "Gentle, soothing voice",
}

_KEY_PATTERN = re.compile(r"AIzaSy[A-Za-z0-9_-]{33}|[a-zA-Z0-9_-]{32,}")


def _sanitize_error_message(msg: str) -> str:
    """Mask credentials from exception messages."""
    return _KEY_PATTERN.sub("[REDACTED_API_KEY]", msg)


def _l16_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """
    Wrap raw L16 (signed 16-bit little-endian PCM) bytes in a RIFF/WAVE header.

    Gemini TTS returns bare PCM samples (MIME: audio/L16;codec=pcm;rate=N).
    Browsers cannot play raw L16 directly; they require a WAVE container.
    """
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    data_size = len(pcm_bytes)
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM subchunk size
        1,  # AudioFormat = PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,  # BitsPerSample
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _parse_l16_sample_rate(mime_type: str) -> int:
    """Extract sample rate from 'audio/L16;codec=pcm;rate=<N>', defaulting to 24000."""
    m = re.search(r"rate=(\d+)", mime_type or "")
    return int(m.group(1)) if m else 24000


def _needs_wav_wrapping(mime_type: str) -> bool:
    """Return True if MIME type indicates raw L16 PCM that needs a WAV header."""
    mt = (mime_type or "").lower()
    return "l16" in mt or ("pcm" in mt and "wav" not in mt)


class GeminiTTSProvider:
    """Production Text-to-Speech provider using Gemini multimodal audio output."""

    provider_name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self._api_key = settings.gemini_api_key.get_secret_value()
        # Prefer configured model; fall back to known-working TTS-specific model
        configured = (settings.tts_model or "").strip()
        self._model = configured if configured else GEMINI_TTS_MODEL
        self._default_voice = settings.tts_voice or "Aoede"
        self._default_language = settings.tts_language_code or "en-IN"
        self._timeout_seconds = settings.llm_timeout_seconds
        # SDK expects timeout in milliseconds
        self._client_timeout_ms = self._timeout_seconds * 1000

        # Don't loop in retries if 429/404 occurs — fail fast and try candidate models
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=self._client_timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _build_tts_config(self, voice: str) -> types.GenerateContentConfig:
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        )
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
            temperature=0.2,
        )

    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResponse:
        t_tts_start = time.perf_counter()
        voice = request.voice_name or self._default_voice
        lang = request.language_code or self._default_language
        config = self._build_tts_config(voice)

        prompt = (
            f"Generate speech for the supplied dialogue in a professional, "
            f"warm, composed customer-support voice. "
            f"Language: {lang}. The dialogue is:\n{request.text}"
        )

        # Candidates to try: primary model first, followed by known fallback models
        candidates = [self._model] + [m for m in TTS_CANDIDATE_MODELS if m != self._model]
        last_error: Exception | None = None
        response = None
        used_model = self._model

        # One total deadline across ALL candidate models (previously each candidate got
        # its own 15s window, so 4 models could burn ~60s). A single attempt may use up
        # to _TTS_ATTEMPT_CAP_S so a slow-but-working synthesis is not cut off at 15s.
        _TTS_ATTEMPT_CAP_S = 25.0
        tts_deadline = time.perf_counter() + float(self._timeout_seconds)
        for model_name in candidates:
            remaining = tts_deadline - time.perf_counter()
            if remaining <= 1.0:
                logger.warning("tts_deadline_exhausted", last_model=model_name)
                break
            try:
                async with asyncio.timeout(min(remaining, _TTS_ATTEMPT_CAP_S)):
                    response = await self._client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                used_model = model_name
                break
            except (asyncio.TimeoutError, TimeoutError) as exc:
                last_error = ProviderTimeoutError(
                    f"Gemini TTS request for '{model_name}' timed out after {self._timeout_seconds}s"
                )
                logger.warning("tts_model_timeout_trying_next", model=model_name)
            except errors.ClientError as exc:
                code = getattr(exc, "code", 0)
                clean_err = _sanitize_error_message(str(exc))
                if code in (404, 429):
                    last_error = ProviderError(
                        f"Gemini TTS model '{model_name}' unavailable (HTTP {code}): {clean_err}"
                    )
                    logger.warning("tts_model_unavailable_trying_next", model=model_name, code=code)
                    continue
                raise ProviderError(f"Gemini TTS client error (HTTP {code}): {clean_err}") from exc
            except Exception as exc:
                clean_err = _sanitize_error_message(str(exc))
                last_error = ProviderError(
                    f"Gemini TTS synthesis failed on '{model_name}': {clean_err}"
                )
                logger.warning("tts_model_error_trying_next", model=model_name, error=clean_err)

        if response is None:
            if last_error:
                raise last_error
            raise ProviderError("Gemini TTS synthesis failed across all available models.")

        audio_bytes = b""
        raw_mime = ""

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    audio_bytes = part.inline_data.data
                    raw_mime = part.inline_data.mime_type or ""
                    break

        if not audio_bytes:
            raise ProviderError("Gemini TTS returned no audio data in response candidate.")

        # Parse actual sample rate from MIME (e.g. "audio/L16;codec=pcm;rate=24000")
        sample_rate = _parse_l16_sample_rate(raw_mime)

        # Wrap raw L16 PCM in a RIFF/WAVE container so browsers can decode it
        if _needs_wav_wrapping(raw_mime):
            audio_bytes = _l16_to_wav(audio_bytes, sample_rate=sample_rate)
            audio_format = "audio/wav"
        else:
            audio_format = raw_mime or "audio/wav"

        words = len(request.text.split())
        est_duration = max(0.5, words / 2.5)
        duration_ms = (time.perf_counter() - t_tts_start) * 1000.0

        logger.info(
            "timing_gemini_tts",
            model=used_model,
            duration_ms=round(duration_ms, 2),
            wav_bytes=len(audio_bytes),
            sample_rate=sample_rate,
            voice=voice,
        )

        logger.info(
            "tts_synthesis_complete",
            model=self._model,
            voice=voice,
            wav_bytes=len(audio_bytes),
            sample_rate=sample_rate,
            est_duration_s=round(est_duration, 2),
        )

        return SpeechSynthesisResponse(
            audio_data=audio_bytes,
            audio_format=audio_format,
            duration_seconds=round(est_duration, 2),
            provider=self.provider_name,
            sample_rate=sample_rate,
        )

    async def synthesize_stream(self, request: SpeechSynthesisRequest) -> AsyncIterator[bytes]:
        """Stream synthesized audio as raw PCM bytes.

        Yields raw L16 PCM sample data as it arrives from the provider.
        The caller is responsible for wrapping in a WAV container if needed.
        """
        voice = request.voice_name or self._default_voice
        config = self._build_tts_config(voice)
        prompt = (
            f"Generate speech for the supplied dialogue in a professional, "
            f"warm customer-support voice. The dialogue is:\n{request.text}"
        )

        try:
            stream_coro = self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=config,
            )
            stream = await stream_coro if inspect.isawaitable(stream_coro) else stream_coro
            async with asyncio.timeout(self._timeout_seconds):
                async for chunk in stream:
                    if chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if getattr(part, "inline_data", None) and part.inline_data.data:
                                yield part.inline_data.data
        except asyncio.TimeoutError as exc:
            raise ProviderTimeoutError("Gemini TTS streaming timed out.") from exc
        except Exception as exc:
            clean_err = _sanitize_error_message(str(exc))
            raise ProviderError(f"Gemini TTS streaming failed: {clean_err}") from exc

    async def synthesize_stream_framed(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancellation_token: Any = None,
    ) -> AsyncIterator[bytes]:
        """Stream synthesized audio as framed PCM chunks suitable for progressive browser playback.

        Yields frames in this order:
          1. Header frame (sample_rate, channels)
          2. PCM data frames
          3. Done frame

        Each frame is packed as: [4-byte big-endian length][1-byte type][payload].
        The browser can start decoding and playing the first PCM frame immediately
        without waiting for the full audio file.

        If a cancellation_token is provided, streaming stops early when the token
        is cancelled (e.g. user barge-in).
        """
        from app.voice.streaming import (
            FRAME_TYPE_DONE,
            FRAME_TYPE_HEADER,
            FRAME_TYPE_PCM,
            pack_done,
            pack_frame,
            pack_header,
            pack_pcm_chunk,
        )

        voice = request.voice_name or self._default_voice
        lang = request.language_code or self._default_language
        config = self._build_tts_config(voice)
        prompt = (
            f"Generate speech for the supplied dialogue. "
            f"Use a professional, warm, composed customer-support voice. "
            f"Language: {lang}. "
            f"The dialogue to speak is:\n"
            f"{request.text}"
        )

        candidates = [self._model] + [m for m in TTS_CANDIDATE_MODELS if m != self._model]
        sample_rate = 24000
        channels = 1
        first_chunk = True
        total_pcm = b""
        last_error: Exception | None = None

        tts_deadline = time.perf_counter() + float(self._timeout_seconds)

        for model_name in candidates:
            remaining = tts_deadline - time.perf_counter()
            if remaining <= 1.0:
                logger.warning("tts_streaming_deadline_exhausted", last_model=model_name)
                break

            try:
                stream_coro = self._client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                stream = await stream_coro if inspect.isawaitable(stream_coro) else stream_coro

                async with asyncio.timeout(min(remaining, 25.0)):
                    async for chunk in stream:
                        if cancellation_token is not None and getattr(
                            cancellation_token, "is_cancelled", False
                        ):
                            logger.info(
                                "tts_stream_cancelled",
                                token_id=getattr(cancellation_token, "token_id", "?"),
                                reason=getattr(cancellation_token, "cancelled_reason", "unknown"),
                            )
                            return

                        if not chunk.candidates:
                            continue
                        for part in chunk.candidates[0].content.parts:
                            if not (getattr(part, "inline_data", None) and part.inline_data.data):
                                continue

                            pcm_data = part.inline_data.data
                            raw_mime = part.inline_data.mime_type or ""

                            if first_chunk:
                                sample_rate = _parse_l16_sample_rate(raw_mime)
                                channels = 1
                                first_chunk = False
                                # Send header first so browser can start decoding
                                yield pack_header(sample_rate, channels)

                            total_pcm += pcm_data
                            yield pack_pcm_chunk(pcm_data)

                if not first_chunk:
                    # Successfully streamed at least one chunk
                    yield pack_done()
                    return
                else:
                    logger.warning("tts_streaming_empty_candidate_trying_next", model=model_name)
                    continue

            except (asyncio.TimeoutError, TimeoutError) as exc:
                last_error = ProviderTimeoutError(
                    f"Gemini TTS streaming on '{model_name}' timed out."
                )
                logger.warning("tts_streaming_timeout_trying_next", model=model_name)
                continue
            except errors.ClientError as exc:
                code = getattr(exc, "code", 0)
                clean_err = _sanitize_error_message(str(exc))
                if code in (404, 429):
                    last_error = ProviderError(
                        f"Gemini TTS model '{model_name}' unavailable (HTTP {code}): {clean_err}"
                    )
                    logger.warning(
                        "tts_streaming_model_unavailable_trying_next", model=model_name, code=code
                    )
                    continue
                last_error = ProviderError(f"Gemini TTS client error (HTTP {code}): {clean_err}")
                break
            except Exception as exc:
                clean_err = _sanitize_error_message(str(exc))
                last_error = ProviderError(
                    f"Gemini TTS streaming failed on '{model_name}': {clean_err}"
                )
                logger.warning("tts_streaming_error_trying_next", model=model_name, error=clean_err)
                continue

        # If all candidates failed
        err_msg = (
            str(last_error)
            if last_error
            else "Gemini TTS streaming failed across all candidate models."
        )
        yield pack_frame(0x05, err_msg.encode("utf-8"))
        if last_error:
            raise last_error
        raise ProviderError("Gemini TTS streaming failed across all available models.")

    async def health_check(self) -> bool:
        try:
            return bool(self._client)
        except Exception:
            return False


class GeminiSTTProvider:
    """Production Speech-to-Text provider using Gemini multimodal audio ingestion."""

    provider_name = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self._api_key = settings.gemini_api_key.get_secret_value()
        configured = (settings.stt_model or "").strip()
        self._model = configured if configured else GEMINI_STT_MODEL
        self._timeout_seconds = settings.llm_timeout_seconds
        self._client_timeout_ms = self._timeout_seconds * 1000

        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=self._client_timeout_ms),
        )

    async def transcribe(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResponse:
        mime = (
            f"audio/{request.audio_format}"
            if not request.audio_format.startswith("audio/")
            else request.audio_format
        )

        audio_part = types.Part.from_bytes(data=request.audio_data, mime_type=mime)
        instruction_part = types.Part.from_text(
            text=(
                f"Transcribe this customer speech accurately and verbatim. "
                f"Language hint: {request.language_hint}. "
                f"If the speaker uses a mix of Indian English and Hindi, transcribe both naturally. "
                f"Output only the transcribed words, without timestamps or commentary."
            )
        )

        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=[audio_part, instruction_part],
                    config=types.GenerateContentConfig(temperature=0.0),
                )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeoutError("Gemini STT request timed out.") from exc
        except Exception as exc:
            clean_err = _sanitize_error_message(str(exc))
            raise ProviderError(f"Gemini STT failed: {clean_err}") from exc

        transcript = (response.text or "").strip()
        return AudioTranscriptionResponse(
            text=transcript,
            confidence=0.95,
            detected_language=request.language_hint,
            provider=self.provider_name,
        )

    async def health_check(self) -> bool:
        return bool(self._client)

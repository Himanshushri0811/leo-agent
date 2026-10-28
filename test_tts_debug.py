import asyncio
from app.core.config import get_settings
from app.voice.gemini_voice import GeminiTTSProvider
from app.ai.interfaces import SpeechSynthesisRequest

async def main():
    settings = get_settings()
    provider = GeminiTTSProvider(settings)

    print("Configured TTS model:", provider._model)
    print("Timeout:", provider._timeout_seconds, "seconds")

    request = SpeechSynthesisRequest(
        text="Hello LEO, this is a short audio test.",
        voice_name="Aoede",
        language_code="en-IN",
    )

    try:
        result = await provider.synthesize(request)
        print("TTS SUCCESS")
        print("Audio bytes:", len(result.audio_data))
        print("Audio format:", result.audio_format)
        print("Provider:", result.provider)
    except Exception as exc:
        print("TTS FAILED:", type(exc).__name__, str(exc))

if __name__ == "__main__":
    asyncio.run(main())

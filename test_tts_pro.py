import asyncio
from app.core.config import get_settings
from app.voice.gemini_voice import GeminiTTSProvider
from app.ai.interfaces import SpeechSynthesisRequest

async def main():
    settings = get_settings()
    provider = GeminiTTSProvider(settings)

    # Test the model directly without trying other fallback models.
    provider._model = "gemini-2.5-pro-preview-tts"

    try:
        result = await provider.synthesize(
            SpeechSynthesisRequest(
                text="Hello LEO, welcome to Bitmax Group.",
                voice_name="Aoede",
                language_code="en-IN",
            )
        )
        print("SUCCESS: TTS audio generated")
        print("Audio bytes:", len(result.audio_data))
        print("Audio format:", result.audio_format)
    except Exception as exc:
        print("FAILED:", type(exc).__name__, str(exc))

if __name__ == "__main__":
    asyncio.run(main())

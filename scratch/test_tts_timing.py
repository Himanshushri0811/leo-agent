import asyncio
import time
from app.core.config import get_settings
from app.ai.interfaces import SpeechSynthesisRequest
from app.voice.gemini_voice import GeminiTTSProvider

async def test_tts():
    settings = get_settings()
    tts = GeminiTTSProvider(settings)

    print(f"TTS Provider model: {tts._model}")
    text = "Thank you for contacting Bitmax Group. I would be happy to explain our cloud architecture consulting services."

    t0 = time.perf_counter()
    req = SpeechSynthesisRequest(
        text=text,
        voice_name="Aoede",
        language_code="en-IN",
    )
    res = await tts.synthesize(req)
    dur = time.perf_counter() - t0
    print(f"TTS Synthesis duration: {dur*1000:.1f}ms, bytes: {len(res.audio_data)}, format: {res.audio_format}")

if __name__ == "__main__":
    asyncio.run(test_tts())

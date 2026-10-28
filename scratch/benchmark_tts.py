import asyncio
import time
from google import genai
from google.genai import types
from app.core.config import get_settings

async def benchmark_tts_models():
    settings = get_settings()
    api_key = settings.gemini_api_key.get_secret_value()
    client = genai.Client(api_key=api_key)

    candidates = [
        "gemini-3.8-flash-tts",
        "gemini-3.8-flash-lite-tts",
        "gemini-3.1-flash-tts-preview",
        "gemini-2.5-flash-preview-tts",
    ]

    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        )
    )
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=speech_config,
        temperature=0.2,
    )

    prompt = "Thank you for calling Bitmax Group. How may I help you today?"

    for model_name in candidates:
        t0 = time.perf_counter()
        try:
            res = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            dur = time.perf_counter() - t0
            bytes_len = 0
            if res.candidates:
                for part in res.candidates[0].content.parts:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        bytes_len = len(part.inline_data.data)
                        break
            print(f"Model {model_name}: {dur*1000:.1f}ms (Audio bytes: {bytes_len})")
        except Exception as e:
            dur = time.perf_counter() - t0
            print(f"Model {model_name} FAILED in {dur*1000:.1f}ms: {type(e).__name__}: {str(e)[:100]}")

if __name__ == "__main__":
    asyncio.run(benchmark_tts_models())

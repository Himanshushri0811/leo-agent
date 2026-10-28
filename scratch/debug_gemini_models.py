import asyncio
from google import genai
from google.genai import types
from app.core.config import get_settings

async def debug_gemini():
    settings = get_settings()
    api_key = settings.gemini_api_key.get_secret_value()
    client = genai.Client(api_key=api_key)

    # List models or test gemini-3.8-flash vs gemini-2.5-flash etc.
    models_to_test = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for m in models_to_test:
        print(f"\nTesting model: {m}")
        try:
            resp = await client.aio.models.generate_content(
                model=m,
                contents="Say 'hello' in one word.",
            )
            print(f"Success with {m}: {resp.text}")
        except Exception as e:
            print(f"Failed with {m}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(debug_gemini())

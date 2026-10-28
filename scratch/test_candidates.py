import asyncio
from google import genai
from app.core.config import get_settings

async def test_candidates():
    settings = get_settings()
    api_key = settings.gemini_api_key.get_secret_value()
    client = genai.Client(api_key=api_key)

    candidates = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro",
    ]

    for m in candidates:
        try:
            r = await client.aio.models.generate_content(model=m, contents="Say 'ok' in one word.")
            print(f"SUCCESS with {m}: {r.text.strip()}")
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err:
                print(f"QUOTA EXHAUSTED for {m}")
            elif "NOT_FOUND" in err or "404" in err:
                print(f"NOT FOUND: {m}")
            else:
                print(f"FAILED for {m}: {type(e).__name__}: {err[:100]}")

if __name__ == "__main__":
    asyncio.run(test_candidates())

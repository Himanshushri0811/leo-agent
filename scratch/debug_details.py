import asyncio
import time
from google import genai
from google.genai import types
from app.core.config import get_settings

async def debug_provider_details():
    settings = get_settings()
    api_key = settings.gemini_api_key.get_secret_value()

    # Test 1: With HttpOptions(timeout=15000)
    print("--- Test 1: HttpOptions(timeout=15000) ---")
    c1 = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=15000))
    t0 = time.perf_counter()
    try:
        r1 = await c1.aio.models.generate_content(model="gemini-3.8-flash", contents="Hi")
        print(f"Test 1 success in {(time.perf_counter()-t0)*1000:.1f}ms: {r1.text}")
    except Exception as e:
        print(f"Test 1 failed in {(time.perf_counter()-t0)*1000:.1f}ms: {type(e).__name__}: {e}")

    # Test 2: With system_instruction and Content objects
    print("\n--- Test 2: With system_instruction and types.Content ---")
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        max_output_tokens=100,
        temperature=0.2,
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hi")])]
    t0 = time.perf_counter()
    try:
        r2 = await c1.aio.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents,
            config=config,
        )
        print(f"Test 2 success in {(time.perf_counter()-t0)*1000:.1f}ms: {r2.text}")
    except Exception as e:
        print(f"Test 2 failed in {(time.perf_counter()-t0)*1000:.1f}ms: {type(e).__name__}: {e}")

    # Test 3: Run the actual GeminiLLMProvider.generate
    print("\n--- Test 3: Actual GeminiLLMProvider ---")
    from app.ai.providers.gemini_provider import GeminiLLMProvider
    from app.ai.interfaces import GenerationRequest, Message
    p = GeminiLLMProvider(settings)
    t0 = time.perf_counter()
    try:
        r3 = await p.generate(GenerationRequest(
            messages=[Message(role="system", content="You are helpful."), Message(role="user", content="Hi")],
            max_tokens=50,
        ))
        print(f"Test 3 success in {(time.perf_counter()-t0)*1000:.1f}ms: {r3.content}")
    except Exception as e:
        print(f"Test 3 failed in {(time.perf_counter()-t0)*1000:.1f}ms: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(debug_provider_details())

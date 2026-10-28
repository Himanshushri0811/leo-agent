import asyncio
import time
from app.ai.interfaces import GenerationRequest, Message
from app.ai.providers.gemini_provider import GeminiLLMProvider
from app.core.config import get_settings

async def test_gemini():
    settings = get_settings()
    print(f"Provider: {settings.llm_provider}")
    print(f"Model in settings: {settings.gemini_model}")
    print(f"Timeout: {settings.llm_timeout_seconds}")
    provider = GeminiLLMProvider(settings)
    print(f"Resolved provider._model: {provider._model}")
    print(f"Resolved client timeout ms: {provider._client_timeout_ms}")

    t0 = time.perf_counter()
    req = GenerationRequest(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello, what is 2+2? Keep it brief."),
        ],
        max_tokens=50,
    )
    try:
        res = await provider.generate(req)
        dur = time.perf_counter() - t0
        print(f"Success in {dur*1000:.1f}ms: {res.content}")
    except Exception as e:
        dur = time.perf_counter() - t0
        print(f"Failed after {dur*1000:.1f}ms: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini())

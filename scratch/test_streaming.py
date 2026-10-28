import asyncio
import time
from app.ai.interfaces import GenerationRequest, Message
from app.ai.providers.gemini_provider import GeminiLLMProvider
from app.core.config import get_settings

async def test_streaming():
    settings = get_settings()
    # Override model to gemini-3.5-flash
    settings.gemini_model = "gemini-3.5-flash"
    provider = GeminiLLMProvider(settings)

    print(f"Testing Gemini streaming with model: {provider._model}...")
    req = GenerationRequest(
        messages=[
            Message(role="system", content="You are a helpful assistant for Bitmax Group."),
            Message(role="user", content="Briefly describe Bitmax Group cloud services in 2 sentences."),
        ],
        max_tokens=150,
    )

    t0 = time.perf_counter()
    first_token_time = None
    chunks = []
    async for chunk in provider.generate_stream(req):
        if first_token_time is None:
            first_token_time = time.perf_counter() - t0
        chunks.append(chunk)

    total_time = time.perf_counter() - t0
    full_text = "".join(chunks)

    print(f"Time to first token (TTFT): {first_token_time*1000:.1f}ms" if first_token_time else "No tokens received")
    print(f"Total generation time: {total_time*1000:.1f}ms")
    print(f"Full text: {full_text}")

if __name__ == "__main__":
    asyncio.run(test_streaming())

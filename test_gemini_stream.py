import asyncio
import time
from app.core.config import get_settings
from app.ai.providers.gemini_provider import GeminiLLMProvider
from app.ai.interfaces import GenerationRequest, Message

async def main():
    settings = get_settings()
    provider = GeminiLLMProvider(settings)

    request = GenerationRequest(
        messages=[
            Message(role="user", content="Reply briefly: Hello from LEO.")
        ],
        max_tokens=50,
        temperature=0,
    )

    start = time.perf_counter()
    chunks = []

    try:
        async for chunk in provider.generate_stream(request):
            chunks.append(chunk)
            print("CHUNK:", repr(chunk))

        print("FULL RESPONSE:", repr("".join(chunks)))
        print("DURATION:", round(time.perf_counter() - start, 2), "seconds")
    except Exception as exc:
        print("ERROR TYPE:", type(exc).__name__)
        print("ERROR:", str(exc))

asyncio.run(main())

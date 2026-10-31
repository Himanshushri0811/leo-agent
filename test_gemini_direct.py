import asyncio
from app.core.config import get_settings
from app.ai.providers.gemini_provider import GeminiLLMProvider
from app.ai.interfaces import GenerationRequest, Message

async def main():
    settings = get_settings()
    provider = GeminiLLMProvider(settings)

    request = GenerationRequest(
        messages=[
            Message(role="user", content="Reply with exactly: LEO Gemini test successful.")
        ],
        max_tokens=50,
        temperature=0,
    )

    try:
        result = await provider.generate(request)
        print("Provider:", result.provider)
        print("Model:", result.model)
        print("Response:", repr(result.content))
    except Exception as exc:
        print("ERROR TYPE:", type(exc).__name__)
        print("ERROR:", str(exc))

asyncio.run(main())

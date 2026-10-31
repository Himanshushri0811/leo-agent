import asyncio
from google import genai
from google.genai import types
from app.core.config import get_settings


async def main():
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    result = await client.aio.models.embed_content(
        model="gemini-embedding-2",
        contents="Bitmax Group enterprise technology services",
        config=types.EmbedContentConfig(
            output_dimensionality=1536,
        ),
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    values = result.embeddings[0].values
    print("Model: gemini-embedding-2")
    print("Dimensions:", len(values))
    print("All finite:", all(map(__import__("math").isfinite, values)))
    assert len(values) == 1536, f"Expected 1536, got {len(values)}"
    print("REAL GEMINI EMBEDDING TEST PASSED")


asyncio.run(main())

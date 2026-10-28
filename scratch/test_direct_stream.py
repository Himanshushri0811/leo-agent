import asyncio
from google import genai
from app.core.config import get_settings

async def main():
    c = genai.Client(api_key=get_settings().gemini_api_key.get_secret_value())
    r = await c.aio.models.generate_content(
        model="gemini-3.5-flash",
        contents="Explain Bitmax Group in 2 sentences.",
    )
    print("Direct non-streaming output:")
    print(r.text)

    print("\nDirect streaming output:")
    stream = await c.aio.models.generate_content_stream(
        model="gemini-3.5-flash",
        contents="Explain Bitmax Group in 2 sentences.",
    )
    async for chunk in stream:
        print("CHUNK:", repr(chunk.text))

if __name__ == "__main__":
    asyncio.run(main())

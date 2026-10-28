import asyncio
from google import genai
from app.core.config import get_settings

async def list_available_models():
    settings = get_settings()
    api_key = settings.gemini_api_key.get_secret_value()
    client = genai.Client(api_key=api_key)

    print("Querying available models for this API key...")
    try:
        pager = client.models.list()
        for m in pager:
            methods = m.supported_actions if hasattr(m, "supported_actions") else ""
            print(f"- {m.name} (display: {getattr(m, 'display_name', '')})")
    except Exception as e:
        print(f"Error listing models: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(list_available_models())

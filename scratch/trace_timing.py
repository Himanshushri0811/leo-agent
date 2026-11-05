import asyncio
import time
from app.core.config import get_settings
from app.services.demo_call_service import get_demo_service, CallActionType

async def main():
    settings = get_settings()
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Gemini Model: {settings.gemini_model}")
    print(f"Embeddings Provider: {settings.embeddings_provider}")
    
    demo_service = get_demo_service(settings)
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)
    print(f"Call created: {call.call_id}")
    
    t0 = time.perf_counter()
    turn = await demo_service.send_customer_message(call.call_id, "What services does Global Projects provide?")
    total = (time.perf_counter() - t0) * 1000
    print(f"send_customer_message finished in {total:.1f}ms")
    
    updated = demo_service.get_call(call.call_id)
    leo_turns = [t for t in updated.transcript if t.role.value == "leo"]
    if leo_turns:
        print(f"LEO response: {leo_turns[-1].text[:100]}...")
        print(f"Metadata: {leo_turns[-1].metadata}")

if __name__ == "__main__":
    asyncio.run(main())


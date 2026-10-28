import asyncio
import time
import httpx

async def benchmark():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=60.0) as client:
        # 1. Simulate a call
        t0 = time.perf_counter()
        sim_res = await client.post("/api/v1/demo/calls/simulate", json={"caller_name": "Test Customer", "auto_answer_delay_sec": 0})
        t_sim = time.perf_counter() - t0
        call = sim_res.json()
        call_id = call["call_id"]
        print(f"Simulate Call ({call_id}): {t_sim*1000:.1f}ms")

        # 2. Answer call
        t0 = time.perf_counter()
        ans_res = await client.post(f"/api/v1/demo/calls/{call_id}/action", json={"action": "answer"})
        t_ans = time.perf_counter() - t0
        print(f"Answer Call: {t_ans*1000:.1f}ms")

        # 3. Customer message turn
        test_queries = [
            "What cloud security and migration services does Bitmax offer?",
            "Can you tell me about Bitmax Group's compliance certifications?",
            "What is your SLA for enterprise cloud consulting?",
        ]

        for query in test_queries:
            print(f"\n--- Testing Query: '{query}' ---")
            t0 = time.perf_counter()
            msg_res = await client.post(
                f"/api/v1/demo/calls/{call_id}/message",
                json={"message": query},
            )
            t_msg = time.perf_counter() - t0
            print(f"POST /calls/{call_id}/message: {t_msg*1000:.1f}ms (Status: {msg_res.status_code})")

            # Get call state to find the LEO turn
            call_state = (await client.get(f"/api/v1/demo/calls/{call_id}")).json()
            leo_turns = [t for t in call_state["transcript"] if t["role"] == "leo"]
            if leo_turns:
                last_leo = leo_turns[-1]
                print(f"LEO Response: {last_leo['text'][:120]}...")
                turn_id = last_leo["id"]

                # 4. Synthesize Audio
                t0_tts = time.perf_counter()
                audio_res = await client.post(f"/api/v1/demo/calls/{call_id}/turns/{turn_id}/audio")
                t_tts = time.perf_counter() - t0_tts
                print(f"POST /turns/{turn_id}/audio: {t_tts*1000:.1f}ms (Status: {audio_res.status_code}, Bytes: {len(audio_res.content)})")
                print(f"TOTAL TURN LATENCY (Message + TTS): {(t_msg + t_tts)*1000:.1f}ms")
            else:
                print("No LEO turn found in transcript!")

if __name__ == "__main__":
    asyncio.run(benchmark())

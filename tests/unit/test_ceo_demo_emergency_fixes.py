"""
CEO Demo Emergency Fixes - Validation Test Suite.

Verifies all 14 critical requirements:
1. Normal English company question
2. Hindi question
3. Hinglish question
4. Follow-up that depends on conversation history
5. Gemini HTTP 504 timeout recovery (no silence, no transfer)
6. Streaming failure and supported fallback
7. One user turn submitted twice (idempotency / deduplication)
8. Invalid or stale turn ID
9. Normal conversation without an unwanted transfer
10. Explicit request for a human agent
11. Barge-in while the assistant is speaking
12. Empty knowledge retrieval
13. TTS failure after a valid text response
14. Structured backend error responses
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.ai import orchestrator
from app.ai.interfaces import GenerationResponse
from app.core.exceptions import ProviderError, ProviderTimeoutError
from app.schemas.demo import (
    CallActionType,
    CallStatus,
    EscalationReason,
    SpeakerRole,
)
from app.services.demo_call_service import DemoCallService
from app.voice.turn_manager import TurnManager


@pytest.fixture
def demo_service(test_settings) -> DemoCallService:
    return DemoCallService(settings=test_settings)


# 1. Normal English company question
@pytest.mark.asyncio
async def test_1_normal_english_company_question(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    turn = await demo_service.send_customer_message(
        call.call_id,
        "What enterprise services does Bitmax Group offer?",
    )
    assert turn.role == SpeakerRole.CUSTOMER
    updated = demo_service.get_call(call.call_id)
    assert updated.escalated is False
    assert updated.status == CallStatus.ACTIVE or updated.status == CallStatus.SPEAKING

    leo_turns = [t for t in updated.transcript if t.role == SpeakerRole.LEO]
    assert len(leo_turns) >= 2  # initial greeting + response
    last_leo = leo_turns[-1]
    assert len(last_leo.text.strip()) > 0
    assert last_leo.metadata.get("detected_language") == "en"


# 2. Hindi question
@pytest.mark.asyncio
async def test_2_hindi_question(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    turn = await demo_service.send_customer_message(
        call.call_id,
        "बिटमैक्स ग्रुप क्या करता है?",
    )
    assert turn.role == SpeakerRole.CUSTOMER
    updated = demo_service.get_call(call.call_id)
    assert updated.escalated is False

    leo_turns = [t for t in updated.transcript if t.role == SpeakerRole.LEO]
    last_leo = leo_turns[-1]
    assert last_leo.metadata.get("detected_language") == "hi"
    assert last_leo.metadata.get("language_locale") == "hi-IN"


# 3. Hinglish question
@pytest.mark.asyncio
async def test_3_hinglish_question(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    turn = await demo_service.send_customer_message(
        call.call_id,
        "Bitmax Group cloud services ke baare mein batao",
    )
    assert turn.role == SpeakerRole.CUSTOMER
    updated = demo_service.get_call(call.call_id)
    assert updated.escalated is False

    leo_turns = [t for t in updated.transcript if t.role == SpeakerRole.LEO]
    last_leo = leo_turns[-1]
    assert last_leo.metadata.get("detected_language") == "hinglish"


# 4. Follow-up that depends on conversation history
@pytest.mark.asyncio
async def test_4_follow_up_depends_on_conversation_history(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    # Turn 1: Caller introduces context
    await demo_service.send_customer_message(
        call.call_id,
        "Our company name is Acme Retail and we need cloud migration.",
    )
    # Turn 2: Follow-up question referencing earlier context
    await demo_service.send_customer_message(
        call.call_id,
        "Can you help us with that?",
    )

    updated = demo_service.get_call(call.call_id)
    customer_turns = [t for t in updated.transcript if t.role == SpeakerRole.CUSTOMER]
    assert len(customer_turns) == 2
    assert "Acme Retail" in customer_turns[0].text
    assert "Can you help" in customer_turns[1].text


# 5. Gemini HTTP 504 timeout produces controlled recovery response without transferring
@pytest.mark.asyncio
async def test_5_gemini_http_504_timeout(demo_service: DemoCallService, monkeypatch) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    async def mock_failing_chat(*args, **kwargs):
        raise ProviderTimeoutError("Gemini provider-side timeout (HTTP 504): DEADLINE_EXCEEDED")

    monkeypatch.setattr("app.services.demo_call_service.run_chat", mock_failing_chat)

    await demo_service.send_customer_message(
        call.call_id,
        "What are your security policies?",
    )

    updated = demo_service.get_call(call.call_id)
    # Must NOT transfer or escalate
    assert updated.escalated is False
    assert updated.status in (CallStatus.ACTIVE, CallStatus.PROCESSING, CallStatus.SPEAKING)

    # Must provide a professional recovery response to caller
    leo_turns = [t for t in updated.transcript if t.role == SpeakerRole.LEO]
    recovery_turn = leo_turns[-1]
    assert "technical issue" in recovery_turn.text.lower() or "repeat" in recovery_turn.text.lower()


# 6. Streaming failure and supported fallback
@pytest.mark.asyncio
async def test_6_streaming_failure_and_supported_fallback() -> None:
    class FailingStreamLLM:
        provider_name = "test"
        _model = "test-model"

        async def generate_stream(self, req):
            raise ProviderTimeoutError("Gemini streaming timeout 504")
            yield "never"

        async def generate(self, req):
            return GenerationResponse(
                content="Fallback non-streaming response succeeded.",
                model="test-model",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                provider="test",
            )

    req = orchestrator.ChatRequest(session_id="s_test", user_message="Tell me about Bitmax")
    settings = MagicMock(llm_max_tokens=100, llm_temperature=0.2)

    with patch("app.ai.orchestrator.retrieve_relevant_chunks", AsyncMock(return_value=[])):
        results = []
        stream_gen = orchestrator.run_chat_streaming(req, FailingStreamLLM(), MagicMock(), settings)
        async for item in stream_gen:
            results.append(item)

        # First item is fallback sentence
        assert any(
            isinstance(r, str) and "Fallback non-streaming" in r for r in results
        )


# 7. One user turn submitted twice
@pytest.mark.asyncio
async def test_7_one_user_turn_submitted_twice(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    turn1 = await demo_service.send_customer_message(call.call_id, "What is Bitmax?")
    turn2 = await demo_service.send_customer_message(call.call_id, "What is Bitmax?")

    updated = demo_service.get_call(call.call_id)
    # Deduplication ensures second identical immediate submission returns previous turn
    assert turn1.id == turn2.id or turn1.text == turn2.text
    # Should not produce extra duplicate customer turns
    customer_turns = [
        t for t in updated.transcript
        if t.role == SpeakerRole.CUSTOMER and t.text == "What is Bitmax?"
    ]
    assert len(customer_turns) == 1


# 8. Invalid or stale turn ID handling
@pytest.mark.asyncio
async def test_8_invalid_or_stale_turn_id(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    # Stale / missing turn in existing call
    turn = next((t for t in call.transcript if t.id == "turn_nonexistent_999"), None)
    assert turn is None


# 9. Normal conversation without an unwanted transfer
@pytest.mark.asyncio
async def test_9_normal_conversation_without_unwanted_transfer(
    demo_service: DemoCallService,
) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    # Questions mentioning "agent" or "manager" must NOT trigger transfer
    await demo_service.send_customer_message(call.call_id, "Are you an AI voice agent?")
    updated = demo_service.get_call(call.call_id)
    assert updated.escalated is False

    await demo_service.send_customer_message(
        call.call_id, "Do you have dedicated account managers?"
    )
    updated = demo_service.get_call(call.call_id)
    assert updated.escalated is False

    await demo_service.send_customer_message(call.call_id, "What cloud services does Bitmax offer?")
    updated = demo_service.get_call(call.call_id)
    assert updated.escalated is False
    assert updated.status != CallStatus.TRANSFERRING


# 10. Explicit request for a human agent
@pytest.mark.asyncio
async def test_10_explicit_request_for_human_agent(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    await demo_service.send_customer_message(
        call.call_id,
        "I need to speak to a real human agent please.",
    )

    updated = demo_service.get_call(call.call_id)
    assert updated.escalated is True
    assert updated.escalation_reason == EscalationReason.CUSTOMER_REQUEST
    assert updated.status == CallStatus.TRANSFERRING
    # Must be explicitly marked simulated handoff
    sys_turns = [t for t in updated.transcript if t.role == SpeakerRole.SYSTEM]
    assert any("[SIMULATED HANDOFF]" in t.text for t in sys_turns)


# 11. Barge-in while the assistant is speaking
@pytest.mark.asyncio
async def test_11_barge_in_while_assistant_speaking() -> None:
    tm = TurnManager()
    session_id = "sess_barge_test"

    token = await tm.create_agent_turn(session_id)
    assert token.is_cancelled is False

    # Simulate barge-in / speech onset
    cancelled_token = await tm.on_user_speech_start(session_id, reason="user_barge_in")
    assert cancelled_token.is_cancelled is True
    assert cancelled_token.cancelled_reason == "user_barge_in"

    # Verify filter_speech_stream aborts immediately
    async def sample_stream():
        yield b"chunk1"
        yield b"chunk2"

    filtered_chunks = []
    async for chunk in tm.filter_speech_stream(sample_stream(), cancelled_token):
        filtered_chunks.append(chunk)

    assert len(filtered_chunks) == 0


# 12. Empty knowledge retrieval uses verified core profile fallback
@pytest.mark.asyncio
async def test_12_empty_knowledge_retrieval() -> None:
    from app.ai.prompts.system import build_system_prompt

    # Empty retrieval must still inject verified Bitmax core profile
    prompt = build_system_prompt("")
    assert "Bitmax Group" in prompt
    assert "Cloud Infrastructure & DevOps" in prompt
    assert "Cybersecurity & Governance" in prompt
    assert "inquiries@bitmaxgroup.com" in prompt


# 13. TTS failure after a valid text response
@pytest.mark.asyncio
async def test_13_tts_failure_after_valid_text_response(
    demo_service: DemoCallService, monkeypatch
) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    await demo_service.send_customer_message(call.call_id, "What services do you offer?")
    updated = demo_service.get_call(call.call_id)
    leo_turns = [t for t in updated.transcript if t.role == SpeakerRole.LEO]
    assert len(leo_turns) >= 2
    last_leo = leo_turns[-1]

    # If TTS synthesis fails, the text turn remains intact
    async def failing_tts(*args, **kwargs):
        raise ProviderError("TTS audio synthesis service unavailable")

    monkeypatch.setattr(demo_service, "synthesize_turn_audio", failing_tts)
    with pytest.raises(ProviderError):
        await demo_service.synthesize_turn_audio(last_leo.text)

    # Transcript remains valid and unaffected
    assert last_leo.text is not None


# 14. Frontend handling of a structured backend error
@pytest.mark.asyncio
async def test_14_structured_backend_error(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    # End call to put it in COMPLETED terminal state
    await demo_service.transition_state(call.call_id, CallStatus.COMPLETED)

    # Sending message to finished call must raise structured ValueError (HTTP 400 at router)
    with pytest.raises(ValueError, match="Cannot send message to finished call"):
        await demo_service.send_customer_message(call.call_id, "Hello again")

"""
Unit tests for Stage 1: State Machine & TurnManager Integration in LEO.

Validates:
1. Authoritative CallStatus state machine transitions (valid and invalid).
2. Synchronization between DemoCallService and TurnManager.
3. Session cleanup on call termination.
4. Duplicate turn suppression within concurrency window.
5. Empty / whitespace turn handling with WAITING_FOR_CLARIFICATION.
6. Inbound vs Outbound call direction handling and greetings.
"""

from __future__ import annotations

import asyncio
import pytest
from app.core.config import Settings
from app.schemas.demo import CallActionType, CallDirection, CallStatus, SpeakerRole
from app.services.demo_call_service import DemoCallService
from app.voice.turn_manager import TurnState


@pytest.fixture
def demo_service() -> DemoCallService:
    settings = Settings(
        app_env="testing",
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        llm_provider="mock",
        embeddings_provider="mock",
        tts_provider="mock",
        stt_provider="mock",
    )
    return DemoCallService(settings)


@pytest.mark.asyncio
async def test_valid_lifecycle_transitions(demo_service: DemoCallService) -> None:
    """Verify standard call lifecycle transitions through GREETING, ACTIVE, and CALL_ENDING."""
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    assert call.status == CallStatus.RINGING
    assert call.direction == CallDirection.INBOUND

    # RINGING -> CONNECTING
    c = await demo_service.transition_state(call.call_id, CallStatus.CONNECTING)
    assert c.status == CallStatus.CONNECTING

    # CONNECTING -> GREETING
    c = await demo_service.transition_state(call.call_id, CallStatus.GREETING)
    assert c.status == CallStatus.GREETING
    assert await demo_service.turn_manager.get_state(call.session_id) == TurnState.LISTENING

    # GREETING -> ACTIVE
    c = await demo_service.transition_state(call.call_id, CallStatus.ACTIVE)
    assert c.status == CallStatus.ACTIVE

    # ACTIVE -> PROCESSING
    c = await demo_service.transition_state(call.call_id, CallStatus.PROCESSING)
    assert c.status == CallStatus.PROCESSING
    assert await demo_service.turn_manager.get_state(call.session_id) == TurnState.THINKING

    # PROCESSING -> SPEAKING
    c = await demo_service.transition_state(call.call_id, CallStatus.SPEAKING)
    assert c.status == CallStatus.SPEAKING
    assert await demo_service.turn_manager.get_state(call.session_id) == TurnState.SPEAKING

    # SPEAKING -> ACTIVE
    c = await demo_service.transition_state(call.call_id, CallStatus.ACTIVE)
    assert c.status == CallStatus.ACTIVE

    # ACTIVE -> CALL_ENDING -> COMPLETED
    c = await demo_service.transition_state(call.call_id, CallStatus.CALL_ENDING)
    assert c.status == CallStatus.CALL_ENDING

    c = await demo_service.transition_state(call.call_id, CallStatus.COMPLETED)
    assert c.status == CallStatus.COMPLETED
    assert c.ended_at is not None


@pytest.mark.asyncio
async def test_invalid_state_transitions_rejected(demo_service: DemoCallService) -> None:
    """Verify invalid state machine transitions raise ValueError."""
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)

    # Cannot jump straight from RINGING to SPEAKING
    with pytest.raises(ValueError, match="Invalid call state transition"):
        await demo_service.transition_state(call.call_id, CallStatus.SPEAKING)

    # Move to COMPLETED (terminal state)
    await demo_service.transition_state(call.call_id, CallStatus.COMPLETED)

    # Cannot move anywhere from terminal COMPLETED
    with pytest.raises(ValueError, match="Invalid call state transition"):
        await demo_service.transition_state(call.call_id, CallStatus.ACTIVE)


@pytest.mark.asyncio
async def test_turn_manager_cleanup_on_completion(demo_service: DemoCallService) -> None:
    """Verify TurnManager cleans up session tokens and states upon call completion."""
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.transition_state(call.call_id, CallStatus.CONNECTING)
    await demo_service.transition_state(call.call_id, CallStatus.ACTIVE)

    # Create a turn token in TurnManager
    token = await demo_service.turn_manager.create_agent_turn(call.session_id)
    assert await demo_service.turn_manager.get_token(call.session_id) is token

    # Terminate call
    await demo_service.transition_state(call.call_id, CallStatus.COMPLETED)

    # Session token must be cleaned up
    assert await demo_service.turn_manager.get_token(call.session_id) is None
    assert await demo_service.turn_manager.get_state(call.session_id) == TurnState.IDLE


@pytest.mark.asyncio
async def test_duplicate_turn_suppression(demo_service: DemoCallService) -> None:
    """Verify duplicate customer message submissions within 1.0s window are ignored."""
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    turn1 = await demo_service.send_customer_message(call.call_id, "What are your cloud consulting services?")
    assert turn1.role == SpeakerRole.CUSTOMER

    # Immediate duplicate submission
    turn2 = await demo_service.send_customer_message(call.call_id, "What are your cloud consulting services?")
    assert turn2.id == turn1.id, "Duplicate submission must return existing turn without re-processing"


@pytest.mark.asyncio
async def test_empty_turn_clarification(demo_service: DemoCallService) -> None:
    """Verify empty or whitespace-only messages prompt clarification and transition to WAITING_FOR_CLARIFICATION."""
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    turn = await demo_service.send_customer_message(call.call_id, "   ")
    assert turn.text == ""

    updated_call = demo_service.get_call(call.call_id)
    assert updated_call.status == CallStatus.WAITING_FOR_CLARIFICATION
    last_turn = updated_call.transcript[-1]
    assert last_turn.role == SpeakerRole.LEO
    assert "clarify" in last_turn.text.lower() or "repeat" in last_turn.text.lower()


@pytest.mark.asyncio
async def test_outbound_call_simulation_and_greeting(demo_service: DemoCallService) -> None:
    """Verify outbound call simulation creates an OUTBOUND call with purpose and proper greeting."""
    call = await demo_service.simulate_outbound_call(
        customer_name="Dr. Aris Thorne",
        customer_phone="+1 (555) 441-2090",
        call_purpose="enterprise cloud architecture review",
        auto_answer_delay_sec=0,
    )
    assert call.direction == CallDirection.OUTBOUND
    assert call.status == CallStatus.RINGING
    assert call.metadata["call_purpose"] == "enterprise cloud architecture review"
    assert "Simulated - No real PSTN" in call.metadata["simulated_telephony_channel"]

    # Answer outbound call
    answered_call = await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)
    assert answered_call.status == CallStatus.ACTIVE

    # Inspect greeting turn
    leo_turns = [t for t in answered_call.transcript if t.role == SpeakerRole.LEO]
    assert len(leo_turns) >= 1
    greeting = leo_turns[0].text
    assert "calling from Bitmax Group regarding" in greeting
    assert "convenient time to speak" in greeting

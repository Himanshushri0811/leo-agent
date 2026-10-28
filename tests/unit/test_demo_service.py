"""
Unit tests for DemoCallService:
- State transitions & validation
- 25 concurrent session isolation
- Duplicate event rejection
- Customer message & AI brain turn
- Escalation rules (human request, low confidence)
- Human agent assignment & release
"""

from __future__ import annotations

import pytest

from app.schemas.demo import (
    AgentStatus,
    CallActionType,
    CallStatus,
    EscalationReason,
    SpeakerRole,
)
from app.services.demo_call_service import DemoCallService


@pytest.fixture
def demo_service(test_settings) -> DemoCallService:
    return DemoCallService(settings=test_settings)


@pytest.mark.asyncio
async def test_simulate_incoming_call(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(
        caller_name="Acme Corp Test",
        caller_phone="+1 (555) 123-4567",
        auto_answer_delay_sec=0,
    )
    assert call.caller_name == "Acme Corp Test"
    assert call.status == CallStatus.RINGING
    assert "***" in call.caller_phone_masked
    assert call.is_simulated is True
    assert call.session_id.startswith("demo_sess_")


@pytest.mark.asyncio
async def test_phone_masking(demo_service: DemoCallService) -> None:
    masked = demo_service.mask_phone("+1 (555) 890-1234")
    assert "890" not in masked
    assert masked.endswith("1234")


@pytest.mark.asyncio
async def test_valid_state_transitions(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    assert call.status == CallStatus.RINGING

    # Ringing -> Connecting
    call = await demo_service.transition_state(call.call_id, CallStatus.CONNECTING)
    assert call.status == CallStatus.CONNECTING

    # Connecting -> Active
    call = await demo_service.transition_state(call.call_id, CallStatus.ACTIVE)
    assert call.status == CallStatus.ACTIVE
    assert call.connected_at is not None

    # Active -> Processing
    call = await demo_service.transition_state(call.call_id, CallStatus.PROCESSING)
    assert call.status == CallStatus.PROCESSING

    # Processing -> Speaking
    call = await demo_service.transition_state(call.call_id, CallStatus.SPEAKING)
    assert call.status == CallStatus.SPEAKING

    # Speaking -> Active
    call = await demo_service.transition_state(call.call_id, CallStatus.ACTIVE)
    assert call.status == CallStatus.ACTIVE

    # Active -> Completed
    call = await demo_service.transition_state(call.call_id, CallStatus.COMPLETED)
    assert call.status == CallStatus.COMPLETED
    assert call.ended_at is not None


@pytest.mark.asyncio
async def test_invalid_state_transition_raises(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    # Ringing directly to Speaking is invalid
    with pytest.raises(ValueError, match="Invalid call state transition"):
        await demo_service.transition_state(call.call_id, CallStatus.SPEAKING)

    # Completed call cannot transition anywhere
    await demo_service.transition_state(call.call_id, CallStatus.COMPLETED)
    with pytest.raises(ValueError, match="Invalid call state transition"):
        await demo_service.transition_state(call.call_id, CallStatus.ACTIVE)


@pytest.mark.asyncio
async def test_concurrency_limit_25(demo_service: DemoCallService) -> None:
    calls = []
    for _ in range(25):
        c = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
        calls.append(c)

    assert len(demo_service.list_calls()) == 25

    # 26th concurrent call must raise ValueError
    with pytest.raises(ValueError, match="Concurrency limit reached"):
        await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)

    # Complete one call to free a slot
    await demo_service.transition_state(calls[0].call_id, CallStatus.COMPLETED)

    # Now 26th call succeeds
    c26 = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    assert c26 is not None


@pytest.mark.asyncio
async def test_duplicate_event_rejection(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    assert demo_service.is_duplicate_event(call.call_id, "evt_101") is False
    assert demo_service.is_duplicate_event(call.call_id, "evt_101") is True


@pytest.mark.asyncio
async def test_call_actions_lifecycle(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)

    # Answer action
    call = await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)
    assert call.status == CallStatus.ACTIVE
    assert any(t.role == SpeakerRole.LEO for t in call.transcript)

    # Mute and unmute
    call = await demo_service.execute_call_action(call.call_id, CallActionType.MUTE)
    assert call.is_muted is True
    call = await demo_service.execute_call_action(call.call_id, CallActionType.UNMUTE)
    assert call.is_muted is False

    # End call
    call = await demo_service.execute_call_action(call.call_id, CallActionType.END)
    assert call.status == CallStatus.COMPLETED


@pytest.mark.asyncio
async def test_customer_turn_and_ai_response(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    turn = await demo_service.send_customer_message(
        call.call_id,
        "What enterprise services do you provide?",
    )
    assert turn.role == SpeakerRole.CUSTOMER
    assert turn.text == "What enterprise services do you provide?"

    # Check that LEO turn was appended
    updated_call = demo_service.get_call(call.call_id)
    assert updated_call is not None
    leo_turns = [t for t in updated_call.transcript if t.role == SpeakerRole.LEO]
    assert len(leo_turns) >= 2  # initial greeting + answer
    last_turn = leo_turns[-1]
    assert "Bitmax" in last_turn.text or "Mock LEO response" in last_turn.text


@pytest.mark.asyncio
async def test_escalation_rule_customer_requests_human(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)

    await demo_service.send_customer_message(
        call.call_id,
        "I need to speak to a real human agent please.",
    )

    updated_call = demo_service.get_call(call.call_id)
    assert updated_call.escalated is True
    assert updated_call.escalation_reason == EscalationReason.CUSTOMER_REQUEST


@pytest.mark.asyncio
async def test_agent_assignment_and_release(demo_service: DemoCallService) -> None:
    call = await demo_service.simulate_incoming_call(auto_answer_delay_sec=0)
    await demo_service.execute_call_action(call.call_id, CallActionType.ANSWER)
    await demo_service.trigger_escalation(call.call_id, EscalationReason.MANUAL)

    agents = demo_service.list_agents()
    available_agent = next(a for a in agents if a.status == AgentStatus.AVAILABLE)

    # Assign agent
    updated = await demo_service.assign_agent(call.call_id, available_agent.agent_id)
    assert updated.assigned_agent_id == available_agent.agent_id
    assert any(t.role == SpeakerRole.AGENT for t in updated.transcript)

    # Completing call frees agent
    await demo_service.transition_state(call.call_id, CallStatus.COMPLETED)
    agents_after = demo_service.list_agents()
    agent_state = next(a for a in agents_after if a.agent_id == available_agent.agent_id)
    assert agent_state.status == AgentStatus.AVAILABLE
    assert agent_state.active_call_id is None

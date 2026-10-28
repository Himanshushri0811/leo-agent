"""
Pydantic schemas for the Demo Call Center Operations Dashboard.
Defines call lifecycle states, agent statuses, events, and metrics.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class CallDirection(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallStatus(StrEnum):
    RINGING = "ringing"
    CONNECTING = "connecting"
    GREETING = "greeting"
    ACTIVE = "active"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    TRANSFERRING = "transferring"
    ERROR_RECOVERY = "error_recovery"
    CALL_ENDING = "call_ending"
    COMPLETED = "completed"
    FAILED = "failed"


class EscalationReason(StrEnum):
    CUSTOMER_REQUEST = "customer_request"
    LOW_CONFIDENCE = "low_confidence"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    RULE_TRIGGER = "rule_trigger"
    MANUAL = "manual"
    SYSTEM_ERROR = "system_error"


class AgentStatus(StrEnum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


class CallActionType(StrEnum):
    ANSWER = "answer"
    REJECT = "reject"
    MUTE = "mute"
    UNMUTE = "unmute"
    END = "end"
    BARGE_IN = "barge_in"
    INTERRUPT = "interrupt"


class SpeakerRole(StrEnum):
    CUSTOMER = "customer"
    LEO = "leo"
    SYSTEM = "system"
    AGENT = "agent"


class TranscriptTurn(BaseModel):
    id: str
    role: SpeakerRole
    text: str
    timestamp: datetime
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float | None = None


class DemoCall(BaseModel):
    call_id: str
    session_id: str
    caller_id: str
    caller_name: str
    caller_phone_masked: str
    status: CallStatus
    is_muted: bool = False
    is_simulated: bool = True
    auto_answer_delay_sec: int = 4
    created_at: datetime
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int = 0
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    escalation_notes: str | None = None
    assigned_agent_id: str | None = None
    assigned_agent_name: str | None = None
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    latest_error: str | None = None
    direction: CallDirection = Field(default=CallDirection.INBOUND)
    turn_state: str = "idle"
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanAgent(BaseModel):
    agent_id: str
    name: str
    department: str
    status: AgentStatus
    active_call_id: str | None = None
    calls_handled_today: int = 0


class SimulateCallRequest(BaseModel):
    caller_name: str | None = None
    caller_phone: str | None = None
    auto_answer_delay_sec: int = Field(default=3, ge=0, le=30)
    initial_message: str | None = None


class SimulateOutboundCallRequest(BaseModel):
    customer_name: str | None = None
    customer_phone: str | None = None
    call_purpose: str = Field(default="account inquiry follow-up", max_length=200)
    auto_answer_delay_sec: int = Field(default=2, ge=0, le=30)


class CallActionRequest(BaseModel):
    action: CallActionType
    reason: str | None = None


class SendCustomerMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class EscalateCallRequest(BaseModel):
    reason: EscalationReason = EscalationReason.MANUAL
    notes: str | None = None


class AssignAgentRequest(BaseModel):
    agent_id: str


class CallMetrics(BaseModel):
    total_calls: int
    active_calls: int
    completed_calls: int
    ai_handled_calls: int
    escalated_calls: int
    system_health: str
    telephony_bridge_status: str = "Unavailable (REST Text Active)"
    concurrency_limit: int = 25
    active_sessions_count: int

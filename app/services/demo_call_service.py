"""
Demo Call Center Service for LEO AI Backend.

Provides:
- 25 concurrent isolated call session state machines.
- Validated lifecycle transitions with duplicate event rejection.
- Live integration with LEO's real RAG orchestrator (Mock / OpenAI / Anthropic).
- Escalation detection rules (customer request, low confidence, manual).
- Human agent queue management with simulated assignment.
- Async SSE event stream broadcaster.
- Sensitive customer data masking.
- Clean separation between demo simulation and production telephony.
- PostgreSQL persistence for calls, transcripts, and handoffs via DemoCallRepository.

MULTI-WORKER DEPLOYMENT LIMITATIONS:
- The in-memory SSE event bus (`self._listeners`) uses `asyncio.Queue` per-process.
  When deploying with multiple Uvicorn/Gunicorn workers, clients connected to Worker A
  will not receive broadcast events produced by Worker B unless backed by an external
  pub/sub message broker such as Redis, RabbitMQ, or PostgreSQL LISTEN/NOTIFY.
- Session concurrency locks (`self._lock`) are local to the process. In a multi-worker
  cluster, global concurrency enforcement requires distributed locking or a central
  database row-count check.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
import random
import re
import time
from typing import Any
import uuid

from app.ai.interfaces import Message
from app.ai.orchestrator import (
    ChatRequest as OrchChatRequest,
    ChatResponse as OrchChatResponse,
    run_chat,
    run_chat_streaming,
)
from app.ai.providers import get_embeddings_provider, get_llm_provider
from app.ai.interfaces import SpeechSynthesisRequest
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.demo import (
    AgentStatus,
    CallActionType,
    CallDirection,
    CallMetrics,
    CallStatus,
    DemoCall,
    EscalationReason,
    HumanAgent,
    SpeakerRole,
    TranscriptTurn,
)
from app.voice.turn_manager import TurnManager, TurnState, UtteranceCancellationToken

logger = get_logger(__name__)

MAX_CONCURRENT_SESSIONS = 25

# Valid state transitions
VALID_TRANSITIONS: dict[CallStatus, set[CallStatus]] = {
    CallStatus.RINGING: {
        CallStatus.CONNECTING,
        CallStatus.GREETING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.CONNECTING: {
        CallStatus.GREETING,
        CallStatus.ACTIVE,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.GREETING: {
        CallStatus.ACTIVE,
        CallStatus.PROCESSING,
        CallStatus.SPEAKING,
        CallStatus.INTERRUPTED,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.ACTIVE: {
        CallStatus.PROCESSING,
        CallStatus.SPEAKING,
        CallStatus.INTERRUPTED,
        CallStatus.WAITING_FOR_CLARIFICATION,
        CallStatus.TRANSFERRING,
        CallStatus.ERROR_RECOVERY,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.PROCESSING: {
        CallStatus.SPEAKING,
        CallStatus.ACTIVE,
        CallStatus.INTERRUPTED,
        CallStatus.WAITING_FOR_CLARIFICATION,
        CallStatus.TRANSFERRING,
        CallStatus.ERROR_RECOVERY,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.SPEAKING: {
        CallStatus.ACTIVE,
        CallStatus.PROCESSING,
        CallStatus.INTERRUPTED,
        CallStatus.WAITING_FOR_CLARIFICATION,
        CallStatus.TRANSFERRING,
        CallStatus.ERROR_RECOVERY,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.INTERRUPTED: {
        CallStatus.ACTIVE,
        CallStatus.PROCESSING,
        CallStatus.WAITING_FOR_CLARIFICATION,
        CallStatus.TRANSFERRING,
        CallStatus.ERROR_RECOVERY,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.WAITING_FOR_CLARIFICATION: {
        CallStatus.ACTIVE,
        CallStatus.PROCESSING,
        CallStatus.INTERRUPTED,
        CallStatus.TRANSFERRING,
        CallStatus.ERROR_RECOVERY,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.TRANSFERRING: {
        CallStatus.ACTIVE,
        CallStatus.PROCESSING,
        CallStatus.SPEAKING,
        CallStatus.ERROR_RECOVERY,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.ERROR_RECOVERY: {
        CallStatus.ACTIVE,
        CallStatus.PROCESSING,
        CallStatus.SPEAKING,
        CallStatus.TRANSFERRING,
        CallStatus.CALL_ENDING,
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.CALL_ENDING: {
        CallStatus.COMPLETED,
        CallStatus.FAILED,
    },
    CallStatus.COMPLETED: set(),  # Terminal state
    CallStatus.FAILED: set(),  # Terminal state
}

# Explicit regex patterns for unambiguous human agent handoff requests
HUMAN_ESCALATION_PATTERNS = [
    re.compile(
        r"\b(speak|talk|transfer|connect|switch|put\s+me\s+through)\s+(to|with)?\s*(a\s+|an\s+)?(human|agent|person|representative|specialist|operator|manager|supervisor)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(want|need|like|demand)\s+to\s+(speak|talk)\s+(to|with)\s*(a\s+|an\s+)?(human|agent|person|representative|specialist|operator|manager|supervisor)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(real|live|actual)\s+(person|human|agent|representative)\b", re.IGNORECASE),
    re.compile(
        r"\b(human\s+agent|human\s+assistance|human\s+help|human\s+support|human\s+specialist)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(transfer\s+to\s+human|connect\s+to\s+human|speak\s+to\s+a\s+human)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(insaan|human|agent|person|manager|customer\s*care)\s*(se)?\s*(baat|connect)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(kisi\s+se\s+baat\s+karni\s+hai|agent\s+chahiye|human\s+chahiye)\b", re.IGNORECASE
    ),
]


def is_explicit_human_request(text: str) -> bool:
    """Return True only if the caller explicitly requested escalation to a human agent."""
    clean = text.strip()
    return any(p.search(clean) for p in HUMAN_ESCALATION_PATTERNS)


# Backward-compatibility alias for tests
ESCALATION_KEYWORDS = [
    "speak to a person",
    "talk to a human",
    "real human",
    "human agent",
    "real person",
    "connect to human",
    "speak to a human",
    "talk to an agent",
]

DEFAULT_AGENTS: list[HumanAgent] = [
    HumanAgent(
        agent_id="agt_sarah_01",
        name="Sarah Jenkins",
        department="Tier-2 Enterprise Support",
        status=AgentStatus.AVAILABLE,
        calls_handled_today=14,
    ),
    HumanAgent(
        agent_id="agt_david_02",
        name="David Chen",
        department="Cloud Solutions Architecture",
        status=AgentStatus.AVAILABLE,
        calls_handled_today=9,
    ),
    HumanAgent(
        agent_id="agt_elena_03",
        name="Elena Rostova",
        department="Security & Compliance Operations",
        status=AgentStatus.BUSY,
        calls_handled_today=18,
    ),
    HumanAgent(
        agent_id="agt_marcus_04",
        name="Marcus Vance",
        department="Billing & Account Governance",
        status=AgentStatus.OFFLINE,
        calls_handled_today=6,
    ),
]

SAMPLE_CALLERS = [
    ("Acme Corp - IT Director", "+1 (555) 349-8291"),
    ("Global Logistics - Ops Lead", "+1 (555) 781-4402"),
    ("Fintech Synergy - VP Eng", "+1 (555) 612-9011"),
    ("Horizon Health - CISO Office", "+1 (555) 233-1899"),
    ("Vertex Retail - Cloud Architect", "+1 (555) 890-5544"),
]


class DemoCallService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = asyncio.Lock()
        self._calls: dict[str, DemoCall] = {}
        self._processed_event_keys: set[str] = set()
        self._agents: dict[str, HumanAgent] = {a.agent_id: a.model_copy() for a in DEFAULT_AGENTS}
        self._listeners: list[asyncio.Queue[dict[str, Any]]] = []
        self._llm = get_llm_provider(self._settings)
        self._embeddings = get_embeddings_provider(self._settings)
        self._auto_answer_tasks: dict[str, asyncio.Task[None]] = {}
        self.turn_manager = TurnManager()
        # TTS provider — lazy-initialised to avoid import errors when not configured
        self._tts = None
        try:
            from app.voice import get_tts_provider

            self._tts = get_tts_provider(self._settings)
            logger.info("tts_provider_initialized", provider=self._tts.provider_name)
        except Exception as exc:
            logger.warning("tts_provider_init_failed", error=str(exc))

    def mask_phone(self, phone: str) -> str:
        """Redact sensitive customer phone number digits."""
        digits = re.sub(r"\D", "", phone)
        if len(digits) >= 10:
            return f"+{digits[:-7]} (***) ***-{digits[-4:]}"
        return "+1 (***) ***-" + phone[-4:] if len(phone) >= 4 else "***"

    async def _persist_call(self, call: DemoCall) -> None:
        """Persist a call record to PostgreSQL via DemoCallRepository."""
        try:
            async with asyncio.timeout(1.5):
                from app.db.session import session_context
                from app.repositories.demo_call_repository import DemoCallRepository

                async with session_context() as session:
                    repo = DemoCallRepository(session)
                    await repo.save_call(call)
        except Exception as exc:
            logger.debug("db_persist_call_skipped", call_id=call.call_id, error=str(exc))

    def _schedule_persist_call(self, call: DemoCall) -> None:
        """Schedule non-blocking background persistence to database."""
        asyncio.create_task(self._persist_call(call.model_copy(deep=True)))

    async def _persist_transcript_turn(
        self, call_id: str, turn: TranscriptTurn, turn_index: int
    ) -> None:
        """Persist a transcript turn to PostgreSQL via DemoCallRepository."""
        try:
            async with asyncio.timeout(1.5):
                from app.db.session import session_context
                from app.repositories.demo_call_repository import DemoCallRepository

                async with session_context() as session:
                    repo = DemoCallRepository(session)
                    await repo.add_transcript_turn(call_id, turn, turn_index)
        except Exception as exc:
            logger.debug("db_persist_transcript_skipped", call_id=call_id, error=str(exc))

    def _schedule_persist_transcript_turn(
        self, call_id: str, turn: TranscriptTurn, turn_index: int
    ) -> None:
        """Schedule non-blocking background persistence of transcript turn."""
        asyncio.create_task(
            self._persist_transcript_turn(call_id, turn.model_copy(deep=True), turn_index)
        )

    async def _persist_handoff(
        self,
        call_id: str,
        agent_id: str,
        agent_name: str,
        department: str,
        reason: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Persist an agent escalation handoff record to PostgreSQL."""
        try:
            async with asyncio.timeout(1.5):
                from app.db.session import session_context
                from app.repositories.demo_call_repository import DemoCallRepository

                async with session_context() as session:
                    repo = DemoCallRepository(session)
                    await repo.record_handoff(
                        call_id=call_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        department=department,
                        reason=reason,
                        notes=notes,
                    )
        except Exception as exc:
            logger.debug("db_persist_handoff_skipped", call_id=call_id, error=str(exc))

    async def _record_transcript_turn(self, call: DemoCall, turn: TranscriptTurn) -> None:
        """Append turn to in-memory transcript under lock and schedule persistence to database."""
        async with self._lock:
            turn_index = len(call.transcript)
            call.transcript.append(turn)
        self._schedule_persist_transcript_turn(call.call_id, turn, turn_index)

    async def load_from_db(self, limit: int = 50) -> int:
        """Re-hydrate in-memory call cache from PostgreSQL on startup or recovery."""
        try:
            from app.db.session import session_context
            from app.repositories.demo_call_repository import DemoCallRepository

            async with session_context() as session:
                repo = DemoCallRepository(session)
                calls = await repo.list_calls(limit=limit)
                async with self._lock:
                    for call in calls:
                        if call.call_id not in self._calls:
                            self._calls[call.call_id] = call
                logger.info("rehydrated_calls_from_db", count=len(calls))
                return len(calls)
        except Exception as exc:
            logger.debug("db_load_calls_skipped", error=str(exc))
            return 0

    async def get_or_load_call(self, call_id: str) -> DemoCall | None:
        """Retrieve a call from cache, or attempt to load from database if missing."""
        call = self.get_call(call_id)
        if call is not None:
            return call
        try:
            from app.db.session import session_context
            from app.repositories.demo_call_repository import DemoCallRepository

            async with session_context() as session:
                repo = DemoCallRepository(session)
                db_call = await repo.get_call(call_id)
                if db_call is not None:
                    async with self._lock:
                        self._calls[call_id] = db_call
                    return db_call
        except Exception as exc:
            logger.debug("db_get_call_failed", call_id=call_id, error=str(exc))
        return None

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to real-time dashboard events via SSE."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._listeners.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)

    async def broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast an event to all connected dashboard SSE clients."""
        payload = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        async with self._lock:
            stale_queues = []
            for q in self._listeners:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    stale_queues.append(q)
            for sq in stale_queues:
                if sq in self._listeners:
                    self._listeners.remove(sq)

    def is_duplicate_event(self, call_id: str, event_key: str) -> bool:
        """Detect and reject duplicate events for session isolation."""
        combined = f"{call_id}:{event_key}"
        if combined in self._processed_event_keys:
            return True
        self._processed_event_keys.add(combined)
        return False

    async def simulate_incoming_call(
        self,
        caller_name: str | None = None,
        caller_phone: str | None = None,
        auto_answer_delay_sec: int = 3,
        initial_message: str | None = None,
    ) -> DemoCall:
        """Create a new incoming call session up to the 25-concurrency limit."""
        async with self._lock:
            active_count = sum(
                1
                for c in self._calls.values()
                if c.status not in (CallStatus.COMPLETED, CallStatus.FAILED)
            )
            if active_count >= MAX_CONCURRENT_SESSIONS:
                raise ValueError(
                    f"Concurrency limit reached: {MAX_CONCURRENT_SESSIONS} active sessions. "
                    "Complete an active session before initiating a new one."
                )

            call_id = f"call_{uuid.uuid4().hex[:10]}"
            session_id = f"demo_sess_{uuid.uuid4().hex[:12]}"

            if not caller_name or not caller_phone:
                sample_name, sample_phone = random.choice(SAMPLE_CALLERS)
                caller_name = caller_name or sample_name
                caller_phone = caller_phone or sample_phone

            masked_phone = self.mask_phone(caller_phone)

            call = DemoCall(
                call_id=call_id,
                session_id=session_id,
                caller_id=f"CR-{random.randint(1000, 9999)}",
                caller_name=caller_name,
                caller_phone_masked=masked_phone,
                direction=CallDirection.INBOUND,
                status=CallStatus.RINGING,
                turn_state="idle",
                is_muted=False,
                is_simulated=True,
                auto_answer_delay_sec=auto_answer_delay_sec,
                created_at=datetime.now(UTC),
                duration_seconds=0,
                escalated=False,
                transcript=[],
                metadata={
                    "direction": "inbound",
                    "simulated_telephony_channel": "SIP/Demo-WebRTC-Mock-Trunk",
                    "audio_codec": "Opus 48kHz (Simulated)",
                    "region": "us-east-1",
                    "initial_message": initial_message
                    or "Hello! I am calling to inquire about Bitmax Group enterprise services.",
                },
            )
            self._calls[call_id] = call

        # Persist to database
        await self._persist_call(call)

        # Broadcast call arrival
        await self.broadcast_event("call_ringing", call.model_dump(mode="json"))

        # Schedule auto-answer if configured
        if auto_answer_delay_sec > 0:
            task = asyncio.create_task(
                self._auto_answer_after_delay(call_id, auto_answer_delay_sec)
            )
            self._auto_answer_tasks[call_id] = task

        return call

    async def simulate_outbound_call(
        self,
        customer_name: str | None = None,
        customer_phone: str | None = None,
        call_purpose: str = "account inquiry follow-up",
        auto_answer_delay_sec: int = 2,
    ) -> DemoCall:
        """Create a simulated outbound call session with explicit stated purpose."""
        async with self._lock:
            active_count = sum(
                1
                for c in self._calls.values()
                if c.status not in (CallStatus.COMPLETED, CallStatus.FAILED)
            )
            if active_count >= MAX_CONCURRENT_SESSIONS:
                raise ValueError(
                    f"Concurrency limit reached: {MAX_CONCURRENT_SESSIONS} active sessions. "
                    "Complete an active session before initiating a new one."
                )

            call_id = f"call_{uuid.uuid4().hex[:10]}"
            session_id = f"demo_sess_{uuid.uuid4().hex[:12]}"

            if not customer_name or not customer_phone:
                sample_name, sample_phone = random.choice(SAMPLE_CALLERS)
                customer_name = customer_name or sample_name
                customer_phone = customer_phone or sample_phone

            masked_phone = self.mask_phone(customer_phone)

            call = DemoCall(
                call_id=call_id,
                session_id=session_id,
                caller_id=f"CR-{random.randint(1000, 9999)}",
                caller_name=customer_name,
                caller_phone_masked=masked_phone,
                direction=CallDirection.OUTBOUND,
                status=CallStatus.RINGING,
                turn_state="idle",
                is_muted=False,
                is_simulated=True,
                auto_answer_delay_sec=auto_answer_delay_sec,
                created_at=datetime.now(UTC),
                duration_seconds=0,
                escalated=False,
                transcript=[],
                metadata={
                    "direction": "outbound",
                    "call_purpose": call_purpose,
                    "simulated_telephony_channel": "SIP/Demo-Outbound-Mock-Trunk (Simulated - No real PSTN)",
                    "audio_codec": "Opus 48kHz (Simulated)",
                    "region": "us-east-1",
                    "initial_message": f"Hello, I understand you are calling regarding {call_purpose}.",
                },
            )
            self._calls[call_id] = call

        # Persist to database
        await self._persist_call(call)

        # Broadcast call arrival
        await self.broadcast_event("call_ringing", call.model_dump(mode="json"))

        # Schedule auto-answer if configured
        if auto_answer_delay_sec > 0:
            task = asyncio.create_task(
                self._auto_answer_after_delay(call_id, auto_answer_delay_sec)
            )
            self._auto_answer_tasks[call_id] = task

        return call

    async def _auto_answer_after_delay(self, call_id: str, delay_sec: int) -> None:
        try:
            await asyncio.sleep(delay_sec)
            async with self._lock:
                call = self._calls.get(call_id)
                if not call or call.status != CallStatus.RINGING:
                    return
            await self.execute_call_action(call_id, CallActionType.ANSWER)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("auto_answer_failed", call_id=call_id, error=str(exc))

    async def transition_state(self, call_id: str, target_status: CallStatus) -> DemoCall:
        """Validate and transition call state machine."""
        async with self._lock:
            call = self._calls.get(call_id)
            if not call:
                raise KeyError(f"Call {call_id} not found")

            current_status = call.status
            if target_status == current_status:
                return call

            valid_targets = VALID_TRANSITIONS.get(current_status, set())
            if target_status not in valid_targets:
                raise ValueError(
                    f"Invalid call state transition: cannot move from '{current_status}' to '{target_status}'"
                )

            call.status = target_status
            if target_status == CallStatus.ACTIVE and not call.connected_at:
                call.connected_at = datetime.now(UTC)
            elif target_status in (CallStatus.COMPLETED, CallStatus.FAILED):
                call.ended_at = datetime.now(UTC)
                if call.connected_at:
                    call.duration_seconds = int((call.ended_at - call.connected_at).total_seconds())

            # Cancel auto answer task if pending
            if target_status in (CallStatus.COMPLETED, CallStatus.FAILED):
                task = self._auto_answer_tasks.pop(call_id, None)
                if task and not task.done():
                    task.cancel()

                # If an agent was assigned, free them up
                if call.assigned_agent_id and call.assigned_agent_id in self._agents:
                    agent = self._agents[call.assigned_agent_id]
                    if agent.active_call_id == call_id:
                        agent.status = AgentStatus.AVAILABLE
                        agent.active_call_id = None
                        agent.calls_handled_today += 1

            # Synchronize TurnManager turn state
            if target_status == CallStatus.PROCESSING:
                await self.turn_manager.set_state(call.session_id, TurnState.THINKING)
            elif target_status == CallStatus.SPEAKING:
                await self.turn_manager.set_state(call.session_id, TurnState.SPEAKING)
            elif target_status == CallStatus.INTERRUPTED:
                await self.turn_manager.set_state(call.session_id, TurnState.INTERRUPTED)
            elif target_status in (
                CallStatus.ACTIVE,
                CallStatus.WAITING_FOR_CLARIFICATION,
                CallStatus.GREETING,
            ):
                await self.turn_manager.set_state(call.session_id, TurnState.LISTENING)
            elif target_status in (CallStatus.COMPLETED, CallStatus.FAILED):
                await self.turn_manager.cleanup_session(call.session_id)

            call.turn_state = str(await self.turn_manager.get_state(call.session_id))

        # Persist updated status & duration (non-blocking)
        self._schedule_persist_call(call)

        await self.broadcast_event(
            "call_state_changed",
            {
                "call_id": call_id,
                "previous_status": current_status,
                "new_status": target_status,
                "call": call.model_dump(mode="json"),
            },
        )
        return call

    async def execute_call_action(
        self,
        call_id: str,
        action: CallActionType,
        reason: str | None = None,
    ) -> DemoCall:
        """Handle dashboard user actions (Answer, Reject, Mute, End)."""
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        # Duplicate action check
        action_key = f"{action}:{call.status}"
        if action not in (CallActionType.MUTE, CallActionType.UNMUTE):
            if self.is_duplicate_event(call_id, action_key):
                logger.warning("duplicate_call_action_ignored", call_id=call_id, action=action)
                return call

        if action == CallActionType.ANSWER:
            if call.status != CallStatus.RINGING:
                raise ValueError(f"Cannot answer call in state {call.status}")
            # Step 1: Connecting
            await self.transition_state(call_id, CallStatus.CONNECTING)
            await asyncio.sleep(0.2)
            # Step 2: Greeting
            await self.transition_state(call_id, CallStatus.GREETING)

            # System connection turn
            init_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.SYSTEM,
                text="Call connected via LEO AI Virtual Agent Gateway (Bitmax Group Voice Mock Bridge).",
                timestamp=datetime.now(UTC),
            )
            await self._record_transcript_turn(call, init_turn)

            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": init_turn.model_dump(mode="json"),
                },
            )

            # Formulate greeting based on call direction
            if call.direction == CallDirection.OUTBOUND:
                purpose = call.metadata.get("call_purpose", "your account inquiry")
                greeting_text = (
                    f"Hello, this is LEO calling from Bitmax Group regarding {purpose}. "
                    "Is this a convenient time to speak?"
                )
            else:
                greeting_text = (
                    "Hello! Welcome to Bitmax Group. I'm LEO, your AI assistant. "
                    "How can I help you today?"
                )

            greeting_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.LEO,
                text=greeting_text,
                timestamp=datetime.now(UTC),
                metadata={"turn_type": "greeting"},
            )

            await self._record_transcript_turn(call, greeting_turn)

            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": greeting_turn.model_dump(mode="json"),
                },
            )

            # Step 3: Transition to ACTIVE listening state
            await self.transition_state(call_id, CallStatus.ACTIVE)

            # Automatically dispatch initial customer inquiry after short pause
            initial_msg = call.metadata.get("initial_message")
            if initial_msg:
                asyncio.create_task(
                    self._dispatch_customer_message_after_delay(call_id, initial_msg, delay=1.2)
                )

        elif action == CallActionType.REJECT:
            if call.status != CallStatus.RINGING:
                raise ValueError(f"Cannot reject call in state {call.status}")
            call.latest_error = reason or "Call rejected by demo operator."
            await self.transition_state(call_id, CallStatus.COMPLETED)

        elif action == CallActionType.MUTE:
            async with self._lock:
                call.is_muted = True
            await self._persist_call(call)
            await self.broadcast_event("call_muted", {"call_id": call_id, "is_muted": True})

        elif action == CallActionType.UNMUTE:
            async with self._lock:
                call.is_muted = False
            await self._persist_call(call)
            await self.broadcast_event("call_muted", {"call_id": call_id, "is_muted": False})

        elif action in (CallActionType.BARGE_IN, CallActionType.INTERRUPT):
            return await self.handle_barge_in(call_id, reason=reason or "user_barge_in")

        elif action == CallActionType.END:
            await self.transition_state(call_id, CallStatus.CALL_ENDING)
            await self.transition_state(call_id, CallStatus.COMPLETED)

        return call

    async def handle_barge_in(self, call_id: str, reason: str = "user_barge_in") -> DemoCall:
        """Handle customer speech onset / barge-in, cancelling in-flight speech and generation."""
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        cancelled_token = await self.turn_manager.on_user_speech_start(
            call.session_id, reason=reason
        )
        if call.status in (CallStatus.SPEAKING, CallStatus.PROCESSING):
            await self.transition_state(call_id, CallStatus.INTERRUPTED)
            asyncio.create_task(self._reset_to_active_after_delay(call_id, delay=0.2))

        await self.broadcast_event(
            "call_interrupted",
            {
                "call_id": call_id,
                "reason": reason,
                "token_id": getattr(cancelled_token, "token_id", None),
            },
        )
        return call

    async def _reset_to_active_after_delay(self, call_id: str, delay: float = 0.6) -> None:
        """Asynchronously transition call state back to ACTIVE without blocking turn response."""
        await asyncio.sleep(delay)
        try:
            call = self.get_call(call_id)
            if call and call.status == CallStatus.SPEAKING:
                await self.transition_state(call_id, CallStatus.ACTIVE)
        except Exception as exc:
            logger.debug("reset_to_active_failed", call_id=call_id, error=str(exc))

    async def _dispatch_customer_message_after_delay(
        self, call_id: str, message: str, delay: float = 1.0
    ) -> None:
        await asyncio.sleep(delay)
        try:
            call = self.get_call(call_id)
            if call and call.status == CallStatus.ACTIVE:
                # Do not auto-dispatch if customer has already sent a message
                customer_turns = [t for t in call.transcript if t.role == SpeakerRole.CUSTOMER]
                if not customer_turns:
                    await self.send_customer_message(call_id, message)
        except Exception as exc:
            logger.error("initial_customer_message_failed", call_id=call_id, error=str(exc))

    async def send_customer_message(self, call_id: str, message_text: str) -> TranscriptTurn:
        """Customer speaks a turn, triggering LEO AI response processing."""
        t_turn_start = time.perf_counter()
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")
        if call.status in (CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.CALL_ENDING):
            raise ValueError(f"Cannot send message to finished call ({call.status})")

        # Duplicate turn submission protection within 1.0s window
        recent_customer_turns = [t for t in call.transcript if t.role == SpeakerRole.CUSTOMER]
        if recent_customer_turns:
            last_turn = recent_customer_turns[-1]
            if last_turn.text.strip().lower() == message_text.strip().lower():
                now = datetime.now(UTC)
                if (now - last_turn.timestamp).total_seconds() < 1.0:
                    logger.warning(
                        "duplicate_customer_turn_ignored", call_id=call_id, text=message_text[:40]
                    )
                    return last_turn

        # Empty / whitespace-only utterance handling
        clean_text = message_text.strip()
        if not clean_text:
            clarify_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.LEO,
                text="I didn't quite catch that. Could you please repeat or clarify your question?",
                timestamp=datetime.now(UTC),
            )
            await self._record_transcript_turn(call, clarify_turn)
            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": clarify_turn.model_dump(mode="json"),
                },
            )
            await self.transition_state(call_id, CallStatus.WAITING_FOR_CLARIFICATION)
            return TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.CUSTOMER,
                text="",
                timestamp=datetime.now(UTC),
            )

        # 1. Append customer turn
        customer_turn = TranscriptTurn(
            id=f"turn_{uuid.uuid4().hex[:8]}",
            role=SpeakerRole.CUSTOMER,
            text=message_text,
            timestamp=datetime.now(UTC),
        )
        await self._record_transcript_turn(call, customer_turn)

        await self.broadcast_event(
            "transcript_turn",
            {
                "call_id": call_id,
                "turn": customer_turn.model_dump(mode="json"),
            },
        )

        # 2. Check escalation rule: customer explicitly requested a human
        if is_explicit_human_request(message_text):
            await self.trigger_escalation(
                call_id=call_id,
                reason=EscalationReason.CUSTOMER_REQUEST,
                notes=f"Customer explicitly requested human assistance: '{message_text[:80]}'",
            )
            # LEO acknowledges escalation
            ack_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.LEO,
                text="Of course. I can't connect calls directly, but you can reach the Bitmax team through their official channels for a direct conversation.",
                timestamp=datetime.now(UTC),
            )
            await self._record_transcript_turn(call, ack_turn)

            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": ack_turn.model_dump(mode="json"),
                },
            )
            return customer_turn

        # 3. Transition to PROCESSING state while LEO RAG brain computes answer
        await self.transition_state(call_id, CallStatus.PROCESSING)

        # Resolve language and per-session context
        from app.voice.language import resolve_response_language

        det_lang, lang_locale = resolve_response_language(message_text, call.metadata)

        # Create TurnManager agent turn token
        agent_token = await self.turn_manager.create_agent_turn(call.session_id)

        # 4. Invoke the real LEO RAG Orchestrator
        try:
            # Build conversation history from recent turns
            history_messages: list[Message] = []
            for t in call.transcript[-8:-1]:
                if t.role == SpeakerRole.CUSTOMER:
                    history_messages.append(Message(role="user", content=t.text))
                elif t.role == SpeakerRole.LEO:
                    history_messages.append(Message(role="assistant", content=t.text))

            orch_req = OrchChatRequest(
                session_id=call.session_id,
                user_message=message_text,
                conversation_history=history_messages,
            )

            # Call real orchestrator with monotonic timing
            t_rag_start = time.perf_counter()
            ai_response = await run_chat(
                request=orch_req,
                llm=self._llm,
                embeddings=self._embeddings,
                settings=self._settings,
            )
            rag_duration_ms = (time.perf_counter() - t_rag_start) * 1000.0

            # Format sources and metadata
            source_dicts = [
                {
                    "chunk_id": s.chunk_id,
                    "source_name": s.source_name,
                    "source_url": s.source_url,
                    "similarity_score": round(s.similarity_score, 4),
                    "content_excerpt": s.content_excerpt,
                }
                for s in ai_response.sources
            ]

            leo_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.LEO,
                text=ai_response.response_text,
                timestamp=datetime.now(UTC),
                sources=source_dicts,
                metadata={
                    "model": ai_response.model,
                    "provider": ai_response.provider,
                    "prompt_tokens": ai_response.prompt_tokens,
                    "completion_tokens": ai_response.completion_tokens,
                    "prompt_version": ai_response.prompt_version,
                    "sources_count": len(ai_response.sources),
                    "retrieval_duration_ms": getattr(ai_response, "retrieval_duration_ms", None),
                    "llm_duration_ms": getattr(ai_response, "llm_duration_ms", None),
                    "ttft_ms": getattr(ai_response, "ttft_ms", None),
                    "rag_duration_ms": round(rag_duration_ms, 2),
                    "detected_language": det_lang.value,
                    "language_locale": lang_locale,
                },
            )

            # 5. Transition to SPEAKING state (acoustic / speech visualization)
            await self.transition_state(call_id, CallStatus.SPEAKING)

            await self._record_transcript_turn(call, leo_turn)

            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": leo_turn.model_dump(mode="json"),
                },
            )

            # Return to ACTIVE state asynchronously so HTTP turn response is not blocked
            asyncio.create_task(self._reset_to_active_after_delay(call_id, delay=0.6))

            turn_duration_ms = (time.perf_counter() - t_turn_start) * 1000.0
            logger.info(
                "timing_demo_customer_turn",
                call_id=call_id,
                customer_turn_id=customer_turn.id,
                leo_turn_id=leo_turn.id,
                total_duration_ms=round(turn_duration_ms, 2),
                rag_duration_ms=round(rag_duration_ms, 2),
                retrieval_duration_ms=getattr(ai_response, "retrieval_duration_ms", None),
                llm_duration_ms=getattr(ai_response, "llm_duration_ms", None),
                ttft_ms=getattr(ai_response, "ttft_ms", None),
            )

            # Complete TurnManager agent turn
            await self.turn_manager.complete_agent_turn(call.session_id, agent_token)

        except Exception as exc:
            logger.error("leo_brain_invocation_failed", call_id=call_id, error=str(exc))
            # Mark TurnManager complete
            await self.turn_manager.complete_agent_turn(call.session_id, agent_token)

            clean_err = str(exc)
            async with self._lock:
                call.latest_error = f"AI Provider Error: {clean_err[:120]}"
            await self._persist_call(call)

            # Provide a polite, localized recovery response rather than leaving silence or transferring
            from app.ai.orchestrator import get_recovery_message

            recovery_text = get_recovery_message(message_text)

            recovery_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.LEO,
                text=recovery_text,
                timestamp=datetime.now(UTC),
                metadata={
                    "is_recovery": True,
                    "error_detail": clean_err[:200],
                    "detected_language": det_lang.value,
                    "language_locale": lang_locale,
                },
            )
            await self._record_transcript_turn(call, recovery_turn)

            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": recovery_turn.model_dump(mode="json"),
                },
            )

            # Transition back to ACTIVE so caller can repeat question or continue naturally
            await self.transition_state(call_id, CallStatus.ACTIVE)

        return customer_turn

    async def trigger_escalation(
        self,
        call_id: str,
        reason: EscalationReason = EscalationReason.MANUAL,
        notes: str | None = None,
    ) -> DemoCall:
        """Escalate call to human agent queue."""
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        prev_status = call.status
        async with self._lock:
            call.escalated = True
            call.escalation_reason = reason
            call.escalation_notes = notes or f"Call escalated due to {reason.value}."

        await self._persist_call(call)
        await self.transition_state(call_id, CallStatus.TRANSFERRING)

        logger.info(
            "call_transfer_triggered",
            call_id=call_id,
            trigger="customer_explicit_request"
            if reason == EscalationReason.CUSTOMER_REQUEST
            else "manual_action",
            reason=reason.value,
            source_function="trigger_escalation",
            previous_state=str(prev_status.value if hasattr(prev_status, "value") else prev_status),
            new_state=str(CallStatus.TRANSFERRING.value),
            is_simulated=True,
        )

        esc_turn = TranscriptTurn(
            id=f"turn_{uuid.uuid4().hex[:8]}",
            role=SpeakerRole.SYSTEM,
            text=f"[SIMULATED HANDOFF] Call routed to human specialist queue. Reason: {call.escalation_reason.value}.",
            timestamp=datetime.now(UTC),
        )
        await self._record_transcript_turn(call, esc_turn)

        await self.broadcast_event(
            "call_escalated",
            {
                "call_id": call_id,
                "reason": reason.value,
                "notes": call.escalation_notes,
                "turn": esc_turn.model_dump(mode="json"),
            },
        )

        return call

    async def assign_agent(self, call_id: str, agent_id: str) -> DemoCall:
        """Assign an available human agent to an escalated call."""
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        async with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                raise KeyError(f"Agent {agent_id} not found")
            if agent.status == AgentStatus.BUSY and agent.active_call_id != call_id:
                raise ValueError(f"Agent {agent.name} is currently busy with another call")

            agent.status = AgentStatus.BUSY
            agent.active_call_id = call_id
            call.assigned_agent_id = agent.agent_id
            call.assigned_agent_name = agent.name

        await self._persist_call(call)
        await self._persist_handoff(
            call_id=call_id,
            agent_id=agent.agent_id,
            agent_name=agent.name,
            department=agent.department,
            reason=str(call.escalation_reason.value) if call.escalation_reason else "manual",
            notes=call.escalation_notes,
        )

        await self.transition_state(call_id, CallStatus.ACTIVE)

        assign_turn = TranscriptTurn(
            id=f"turn_{uuid.uuid4().hex[:8]}",
            role=SpeakerRole.AGENT,
            text=f"Hello, this is {agent.name} from {agent.department}. I have reviewed your conversation context with LEO and am taking over.",
            timestamp=datetime.now(UTC),
        )
        await self._record_transcript_turn(call, assign_turn)

        await self.broadcast_event(
            "agent_assigned",
            {
                "call_id": call_id,
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "turn": assign_turn.model_dump(mode="json"),
            },
        )

        return call

    def get_call(self, call_id: str) -> DemoCall | None:
        return self._calls.get(call_id)

    def list_calls(
        self,
        status_filter: str | None = None,
        search_query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DemoCall]:
        """List and search calls with pagination."""
        calls = list(self._calls.values())
        # Sort newest first
        calls.sort(key=lambda c: c.created_at, reverse=True)

        if status_filter:
            calls = [c for c in calls if c.status.value == status_filter]

        if search_query:
            q = search_query.lower()
            calls = [
                c
                for c in calls
                if q in c.caller_name.lower()
                or q in c.caller_id.lower()
                or q in c.caller_phone_masked.lower()
                or any(q in t.text.lower() for t in c.transcript)
            ]

        return calls[offset : offset + limit]

    def list_agents(self) -> list[HumanAgent]:
        return list(self._agents.values())

    def get_metrics(self) -> CallMetrics:
        """Aggregate real-time metrics across all calls."""
        total = len(self._calls)
        active = sum(
            1
            for c in self._calls.values()
            if c.status not in (CallStatus.COMPLETED, CallStatus.FAILED)
        )
        completed = sum(1 for c in self._calls.values() if c.status == CallStatus.COMPLETED)
        escalated = sum(1 for c in self._calls.values() if c.escalated)
        ai_handled = sum(
            1 for c in self._calls.values() if not c.escalated and c.status == CallStatus.COMPLETED
        )

        return CallMetrics(
            total_calls=total,
            active_calls=active,
            completed_calls=completed,
            ai_handled_calls=ai_handled,
            escalated_calls=escalated,
            system_health="Operational",
            telephony_bridge_status="Unavailable (REST Text Active)",
            concurrency_limit=MAX_CONCURRENT_SESSIONS,
            active_sessions_count=active,
        )

    async def synthesize_turn_audio(self, text: str, language_code: str | None = None) -> bytes:
        """
        Synthesize speech for a LEO response and return WAV-format audio bytes.

        Uses the configured TTS provider (Gemini by default with Aoede voice).
        Returns a valid RIFF/WAVE file that browsers can play directly.
        Raises ProviderError or ProviderTimeoutError on failure.
        """
        if self._tts is None:
            raise RuntimeError(
                "TTS provider is not configured. "
                "Set TTS_PROVIDER=gemini (or mock) in your .env to enable audio synthesis."
            )

        req = SpeechSynthesisRequest(
            text=text,
            voice_name=self._settings.tts_voice or "Aoede",
            language_code=language_code or self._settings.tts_language_code or "en-IN",
        )
        t_tts_start = time.perf_counter()
        response = await self._tts.synthesize(req)
        tts_dur_ms = (time.perf_counter() - t_tts_start) * 1000.0
        logger.info(
            "timing_demo_service_tts",
            duration_ms=round(tts_dur_ms, 2),
            wav_bytes=len(response.audio_data),
        )
        return response.audio_data

    async def stream_turn_audio_framed(
        self,
        text: str,
        *,
        turn_id: str | None = None,
        cancellation_token: Any = None,
        language_code: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio as framed PCM chunks for progressive browser playback.

        Yields framed binary frames (header, PCM chunks, done) that the browser
        can decode and play incrementally. Falls back to the non-streaming
        implementation if the provider does not support streaming.
        """
        if self._tts is None:
            raise RuntimeError(
                "TTS provider is not configured. "
                "Set TTS_PROVIDER=gemini (or mock) in your .env to enable audio synthesis."
            )

        req = SpeechSynthesisRequest(
            text=text,
            voice_name=self._settings.tts_voice or "Aoede",
            language_code=language_code or self._settings.tts_language_code or "en-IN",
        )
        t_tts_start = time.perf_counter()

        # Try streaming first; fall back to non-streaming if unsupported
        stream_method = getattr(self._tts, "synthesize_stream_framed", None)
        if stream_method is not None:
            try:
                first_byte_ms = None
                async for chunk in stream_method(req, cancellation_token=cancellation_token):
                    if first_byte_ms is None:
                        first_byte_ms = (time.perf_counter() - t_tts_start) * 1000.0
                    yield chunk
                tts_dur_ms = (time.perf_counter() - t_tts_start) * 1000.0
                logger.info(
                    "timing_demo_service_tts_stream",
                    turn_id=turn_id,
                    duration_ms=round(tts_dur_ms, 2),
                    first_audio_byte_ms=round(first_byte_ms, 2) if first_byte_ms else None,
                )
                return
            except (AttributeError, TypeError):
                pass  # Provider doesn't support streaming; fall through

        # Fallback: synthesize full audio and stream as framed chunks
        response = await self._tts.synthesize(req)
        tts_dur_ms = (time.perf_counter() - t_tts_start) * 1000.0
        logger.info(
            "timing_demo_service_tts_fallback",
            turn_id=turn_id,
            duration_ms=round(tts_dur_ms, 2),
            wav_bytes=len(response.audio_data),
        )
        # Stream as framed WAV chunks (non-progressive but compatible)
        from app.voice.streaming import (
            FRAME_TYPE_DONE,
            FRAME_TYPE_HEADER,
            FRAME_TYPE_PCM,
            pack_done,
            pack_frame,
            pack_header,
            pack_pcm_chunk,
        )

        yield pack_header(response.sample_rate, 1)
        chunk_size = 8192
        for i in range(0, len(response.audio_data), chunk_size):
            yield pack_pcm_chunk(response.audio_data[i : i + chunk_size])
        yield pack_done()

    async def stream_llm_then_tts_framed(
        self,
        call_id: str,
        message_text: str,
        *,
        cancellation_token: Any = None,
        language_code: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Full LLM\u2192TTS pipelining: stream LLM tokens, accumulate sentences, synthesize TTS.

        This method pipelines the LLM response through TTS synthesis sentence-by-sentence,
        yielding PCM audio frames as soon as the first sentence is ready from the LLM.
        It significantly reduces time-to-first-audio vs waiting for the full LLM response.

        Yields:
          - FRAME_TYPE_TEXT frames for each sentence (0x03): transcript text chunk
          - FRAME_TYPE_HEADER (0x01): audio sample rate / channels
          - FRAME_TYPE_PCM (0x02): raw PCM audio chunks
          - FRAME_TYPE_DONE (0x04): end of stream
        """
        if self._tts is None:
            raise RuntimeError(
                "TTS provider is not configured. "
                "Set TTS_PROVIDER=gemini (or mock) in your .env to enable audio synthesis."
            )

        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        from app.ai.interfaces import SpeechSynthesisRequest
        from app.voice.streaming import (
            pack_done,
            pack_frame,
            pack_header,
            pack_pcm_chunk,
            pack_text_chunk,
            FRAME_TYPE_TEXT,
        )

        t_pipeline_start = time.perf_counter()

        # Build conversation history
        history_messages: list[Message] = []
        for t in call.transcript[-8:]:
            if t.metadata.get("turn_type") == "greeting":
                continue

            if t.role == SpeakerRole.CUSTOMER:
                history_messages.append(Message(role="user", content=t.text))
            elif t.role == SpeakerRole.LEO:
                history_messages.append(Message(role="assistant", content=t.text))

        # Record customer turn in transcript if not already recorded
        recent_customer = [t for t in call.transcript if t.role == SpeakerRole.CUSTOMER]
        if (
            not recent_customer
            or recent_customer[-1].text.strip().lower() != message_text.strip().lower()
        ):
            cust_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.CUSTOMER,
                text=message_text,
                timestamp=datetime.now(UTC),
            )
            await self._record_transcript_turn(call, cust_turn)
            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": cust_turn.model_dump(mode="json"),
                },
            )

        await self.transition_state(call_id, CallStatus.PROCESSING)

        orch_req = OrchChatRequest(
            session_id=call.session_id,
            user_message=message_text,
            conversation_history=history_messages,
        )

        req = SpeechSynthesisRequest(
            text="",  # Will be overridden per sentence
            voice_name=self._settings.tts_voice or "Aoede",
            language_code=language_code or self._settings.tts_language_code or "en-IN",
        )

        stream_method = getattr(self._tts, "synthesize_stream_framed", None)
        full_response_text: list[str] = []
        header_sent = False
        first_audio_ms: float | None = None
        chat_response: OrchChatResponse | None = None

        try:
            async for item in run_chat_streaming(
                request=orch_req,
                llm=self._llm,
                embeddings=self._embeddings,
                settings=self._settings,
            ):
                # Check cancellation
                if cancellation_token is not None and getattr(
                    cancellation_token, "is_cancelled", False
                ):
                    logger.info("llm_tts_pipeline_cancelled", call_id=call_id)
                    return

                if isinstance(item, OrchChatResponse):
                    # Final metadata item — store it, do not yield audio
                    chat_response = item
                    continue

                # item is a sentence string
                sentence: str = item
                if not sentence.strip():
                    continue

                full_response_text.append(sentence)

                # Transition to SPEAKING when first text is ready
                if len(full_response_text) == 1:
                    try:
                        await self.transition_state(call_id, CallStatus.SPEAKING)
                    except Exception:
                        pass

                # Yield text frame so client can show transcript incrementally
                yield pack_text_chunk(sentence)

                # Synthesize TTS for this sentence
                if stream_method is not None:
                    req_s = SpeechSynthesisRequest(
                        text=sentence,
                        voice_name=req.voice_name,
                        language_code=req.language_code,
                    )
                    try:
                        async for audio_frame in stream_method(
                            req_s, cancellation_token=cancellation_token
                        ):
                            if cancellation_token is not None and getattr(
                                cancellation_token, "is_cancelled", False
                            ):
                                return
                            # The first frame from TTS is always the header — forward it only once
                            if not header_sent and len(audio_frame) >= 5 and audio_frame[4] == 0x01:
                                header_sent = True
                                yield audio_frame
                            elif len(audio_frame) >= 5 and audio_frame[4] == 0x01:
                                # Skip duplicate headers from subsequent sentences
                                continue
                            else:
                                if (
                                    first_audio_ms is None
                                    and len(audio_frame) >= 5
                                    and audio_frame[4] == 0x02
                                ):
                                    first_audio_ms = (
                                        time.perf_counter() - t_pipeline_start
                                    ) * 1000.0
                                    logger.info(
                                        "timing_pipeline_first_audio_byte",
                                        call_id=call_id,
                                        first_audio_ms=round(first_audio_ms, 2),
                                    )
                                if len(audio_frame) >= 5 and audio_frame[4] != 0x04:
                                    # Skip intermediate done frames; yield real PCM
                                    yield audio_frame
                    except Exception as tts_exc:
                        logger.warning(
                            "pipeline_tts_sentence_failed",
                            call_id=call_id,
                            sentence_len=len(sentence),
                            error=str(tts_exc),
                        )

        except Exception as exc:
            logger.error("stream_llm_then_tts_failed", call_id=call_id, error=str(exc))
            raise

        yield pack_done()

        total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        # Record completed LEO turn in call transcript if not already added
        if full_response_text:
            assembled_answer = " ".join(full_response_text).strip()
            recent_leo = [t for t in call.transcript if t.role == SpeakerRole.LEO]
            if not recent_leo or recent_leo[-1].text.strip() != assembled_answer:
                leo_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.LEO,
                    text=assembled_answer,
                    timestamp=datetime.now(UTC),
                    metadata={
                        "streamed": True,
                        "pipeline_total_ms": round(total_pipeline_ms, 2),
                        "first_audio_ms": round(first_audio_ms, 2) if first_audio_ms else None,
                    },
                )
                await self._record_transcript_turn(call, leo_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {
                        "call_id": call_id,
                        "turn": leo_turn.model_dump(mode="json"),
                    },
                )

        try:
            await self.transition_state(call_id, CallStatus.ACTIVE)
        except Exception:
            pass

        logger.info(
            "timing_llm_tts_pipeline_complete",
            call_id=call_id,
            total_ms=round(total_pipeline_ms, 2),
            first_audio_ms=round(first_audio_ms, 2) if first_audio_ms else None,
            sentences_count=len(full_response_text),
        )

        return

        # Yield the assembled chat_response for callers who need it
        # (unreachable in streaming context but accessible via attribute after exhaustion)


_demo_service_instance: DemoCallService | None = None


def get_demo_service() -> DemoCallService:
    global _demo_service_instance
    if _demo_service_instance is None:
        _demo_service_instance = DemoCallService()
    return _demo_service_instance

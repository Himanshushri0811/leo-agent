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
        agent_id="agt_ananya_01",
        name="Ananya Sharma",
        department="Tier-2 Telecom & Enterprise Support",
        status=AgentStatus.AVAILABLE,
        calls_handled_today=14,
    ),
    HumanAgent(
        agent_id="agt_rajesh_02",
        name="Rajesh Verma",
        department="Enterprise Architecture & Systems",
        status=AgentStatus.AVAILABLE,
        calls_handled_today=11,
    ),
    HumanAgent(
        agent_id="agt_priya_03",
        name="Priya Nair",
        department="Customer Escalation & Solutions",
        status=AgentStatus.AVAILABLE,
        calls_handled_today=8,
    ),
    HumanAgent(
        agent_id="agt_david_04",
        name="David Chen",
        department="Cloud Infrastructure Operations",
        status=AgentStatus.BUSY,
        calls_handled_today=19,
    ),
]


SAMPLE_CALLERS = [
    ("Acme Corp - IT Director", "+91 (982) 014-4829"),
    ("Global Logistics - Ops Lead", "+91 (944) 812-9304"),
    ("Fintech Synergy - VP Eng", "+91 (998) 765-4321"),
    ("Horizon Health - CISO Office", "+91 (916) 723-9810"),
    ("Vertex Retail - Cloud Architect", "+91 (983) 345-6789"),
]

SUPPORTED_DEMO_LANGUAGES: list[dict[str, Any]] = [
    {
        "id": "en",
        "key": "1",
        "code": "en-IN",
        "name": "English",
        "native_name": "English",
        "script": "Latin",
        "is_first_level": True,
        "greeting": "English confirmed. Hello! I am LEO, your AI voice assistant. How may I help you today?",
        "tts_verified": True,
        "asr_verified": True,
        "sample_inquiries": [
            "What enterprise cloud services do you provide?",
            "Tell me about Global Projects architecture consulting.",
            "What are your SLA and technical support tiers?",
            "I need to speak with a human agent right now.",
        ],
    },
    {
        "id": "hi",
        "key": "2",
        "code": "hi-IN",
        "name": "Hindi",
        "native_name": "हिन्दी",
        "script": "Devanagari",
        "is_first_level": True,
        "greeting": "हिन्दी चुनी गई। नमस्ते! मैं LEO हूँ, आपका AI वॉइस असिस्टेंट। बताइए, मैं आपकी किस तरह मदद कर सकता हूँ?",
        "tts_verified": True,
        "asr_verified": True,
        "sample_inquiries": [
            "आप कौन सी क्लाउड और आईटी सेवाएं प्रदान करते हैं?",
            "ग्लोबल प्रोजेक्ट्स के बारे में विस्तार से बताएं।",
            "कस्टम सॉफ्टवेयर और एंटरप्राइज समाधान क्या हैं?",
            "मुझे एक मानव विशेषज्ञ से बात करनी है।",
        ],
    },
    {
        "id": "ta",
        "key": "1",
        "code": "ta-IN",
        "name": "Tamil",
        "native_name": "தமிழ்",
        "script": "Tamil",
        "is_first_level": False,
        "greeting": "தமிழ் உறுதிப்படுத்தப்பட்டது. வணக்கம்! நான் LEO, உங்கள் AI உதவியாளர். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "tts_verified": True,
        "asr_verified": True,
        "sample_inquiries": [
            "உங்கள் கிளவுட் சேவைகள் என்னென்ன?",
            "குளோபல் ப்ராஜெக்ட்ஸ் நிறுவனத்தின் சேவைகள் பற்றி கூறுங்கள்.",
            "ஆதரவு மற்றும் பராமரிப்பு விவரங்கள் என்ன?",
            "நான் மனித அதிகாரியுடன் பேச வேண்டும்.",
        ],
    },
    {
        "id": "te",
        "key": "2",
        "code": "te-IN",
        "name": "Telugu",
        "native_name": "తెలుగు",
        "script": "Telugu",
        "is_first_level": False,
        "greeting": "తెలుగు నిర్ధారించబడింది. నమస్కారం! నేను LEO, మీ AI సహాయకుడిని. ఈరోజు మీకు ఎలా సహాయం చేయగలను?",
        "tts_verified": True,
        "asr_verified": True,
        "sample_inquiries": [
            "మీరు అందించే క్లౌڈ సేవల గురించి చెప్పండి.",
            "గ్లోబల్ ప్రాజెక్ట్స్ గురించి వివరాలు కావాలి.",
            "కస్టమర్ సపోర్ట్ ప్లాన్లు ఏమిటి?",
            "నేను మానవ నిపుణుడితో మాట్లాడాలి.",
        ],
    },
    {
        "id": "kn",
        "key": "3",
        "code": "kn-IN",
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "script": "Kannada",
        "is_first_level": False,
        "greeting": "ಕನ್ನಡ ದೃಢೀಕರಿಸಲಾಗಿದೆ. ನಮಸ್ಕಾರ! ನಾನು LEO, ನಿಮ್ಮ AI ಸಹಾಯಕ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
        "tts_verified": True,
        "asr_verified": True,
        "sample_inquiries": [
            "ನಿಮ್ಮ ಕ್ಲೌಡ್ ಸೇವೆಗಳು ಯಾವುವು?",
            "ಗ್ಲೋಬಲ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಿ.",
            "ತಾಂತ್ರಿಕ ಬೆಂಬಲ ಲಭ್ಯವಿದೆಯೇ?",
            "ನಾನು ಅಧಿಕಾರಿಯೊಂದಿಗೆ ಮಾತನಾಡಬೇಕು.",
        ],
    },
    {
        "id": "ml",
        "key": "4",
        "code": "ml-IN",
        "name": "Malayalam",
        "native_name": "മലയാളം",
        "script": "Malayalam",
        "is_first_level": False,
        "greeting": "മലയാളം സ്ഥിരീകരിച്ചു. നമസ്കാരം! ഞാൻ LEO, നിങ്ങളുടെ AI അസിസ്റ്റന്റാണ്. ഇന്ന് നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",
        "tts_verified": True,
        "asr_verified": True,
        "sample_inquiries": [
            "നിങ്ങൾ നൽകുന്ന ക്ലൗഡ് സേവനങ്ങൾ എന്തൊക്കെയാണ്?",
            "ഗ്ലോബൽ പ്രോജക്ട്സ് സേവനങ്ങളെക്കുറിച്ച് വ്യക്തമാക്കുക.",
            "സാങ്കേതിക സഹായം ലഭ്യമാണോ?",
            "എനിക്ക് ഒരു ഏജന്റുമായി സംസാരിക്കണം.",
        ],
    },
    {
        "id": "tcy",
        "key": "5",
        "code": "tcy-IN",
        "name": "Tulu",
        "native_name": "ತುಳು",
        "script": "Kannada / Tulu",
        "is_first_level": False,
        "greeting": "ತುಳು ಆಯ್ಕೆ ಆತ್ಂಡ್. ನಮಸ್ಕಾರ! ಯಾನ್ LEO, ಈರೆನ AI ಸಹಾಯಕ. ಈ ದಿನ ಈರೆಗ್ ಎಂಚಿನ ಸಾಯ ಮಲ್ಪೊಲಿ?",
        "tts_verified": False,
        "asr_verified": False,
        "sample_inquiries": [
            "ಈರೆನ ಕ್ಲೌಡ್ ಸೇವೆಲು ಒವ್ವು?",
            "ಗ್ಲೋಬಲ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಬಗ್ಗೆ ತೆರಿಪಾಲೆ.",
        ],
    },
    {
        "id": "kok",
        "key": "6",
        "code": "kok-IN",
        "name": "Konkani",
        "native_name": "कोंकणी",
        "script": "Devanagari / Roman",
        "is_first_level": False,
        "greeting": "कोंकणी निवडली. देव बरें करूं! हांव LEO, तुमचो AI व्हॉइस असिस्टंट. आज हांव तुमकां कशी मदत करूं?",
        "tts_verified": False,
        "asr_verified": False,
        "sample_inquiries": [
            "तुमच्यो क्लाउड सेवा खंच्यो आसात?",
            "ग्लोबल प्रोजेक्ट्स विशीं म्हायती दिया.",
        ],
    },
    {
        "id": "ur",
        "key": "7",
        "code": "ur-IN",
        "name": "Urdu",
        "native_name": "اردو",
        "script": "Perso-Arabic",
        "is_first_level": False,
        "greeting": "اردو منتخب کی گئی۔ آداب! میں LEO ہوں، آپ کا AI وائس اسسٹنٹ۔ آج میں آپ کی کیا مدد کر سکتا ہوں؟",
        "tts_verified": True,
        "asr_verified": True,
        "sample_inquiries": [
            "آپ کون سی کلاؤڈ اور آئی ٹی خدمات فراہم کرتے हैं؟",
            "گلوبل پروجیکٹس کی سروسز کے بارے میں بتائیں۔",
            "مجھے کسٹمر کیئر ایجنٹ سے بات کرنی ہے۔",
        ],
    },
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
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._db_available: bool = True

        self.turn_manager = TurnManager()
        self._audio_cache: dict[str, list[bytes]] = {}

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

    async def await_pending_persists(self, timeout: float = 5.0) -> None:
        """Wait for in-flight background persistence tasks to complete."""
        if self._background_tasks:
            pending = list(self._background_tasks)
            try:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
            except Exception:
                pass

    async def _persist_call(self, call: DemoCall) -> None:
        """Persist a call record to PostgreSQL via DemoCallRepository."""
        if not getattr(self, "_db_available", True) or self._settings.is_testing:
            return

        try:
            async with asyncio.timeout(1.5):
                from app.db.session import session_context
                from app.repositories.demo_call_repository import DemoCallRepository

                async with session_context() as session:
                    repo = DemoCallRepository(session)
                    await repo.save_call(call)
        except Exception as exc:
            self._db_available = False
            logger.warning("db_persist_call_failed_fallback_in_memory", call_id=call.call_id, error=str(exc))

    def _schedule_persist_call(self, call: DemoCall) -> None:
        """Schedule non-blocking background persistence to database."""
        if not getattr(self, "_db_available", True) or self._settings.is_testing:
            return

        async def _wrapper() -> None:
            try:
                await self._persist_call(call.model_copy(deep=True))
            except Exception as exc:
                self._db_available = False
                logger.warning("db_persist_call_background_error", call_id=call.call_id, error=str(exc))

        task = asyncio.create_task(_wrapper())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _persist_transcript_turn(
        self, call_id: str, turn: TranscriptTurn, turn_index: int
    ) -> None:
        """Persist a transcript turn to PostgreSQL via DemoCallRepository."""
        if not getattr(self, "_db_available", True) or self._settings.is_testing:
            return

        try:
            async with asyncio.timeout(1.5):
                from app.db.session import session_context
                from app.repositories.demo_call_repository import DemoCallRepository

                async with session_context() as session:
                    repo = DemoCallRepository(session)
                    call_rec = await repo.get_call_record(call_id)
                    if call_rec is None and call_id in self._calls:
                        await repo.save_call(self._calls[call_id])
                    await repo.add_transcript_turn(call_id, turn, turn_index)
        except Exception as exc:
            self._db_available = False
            logger.warning("db_persist_transcript_failed_fallback_in_memory", call_id=call_id, error=str(exc))

    def _schedule_persist_transcript_turn(
        self, call_id: str, turn: TranscriptTurn, turn_index: int
    ) -> None:
        """Schedule non-blocking background persistence of transcript turn."""
        if not getattr(self, "_db_available", True) or self._settings.is_testing:
            return

        async def _wrapper() -> None:
            try:
                await self._persist_transcript_turn(call_id, turn.model_copy(deep=True), turn_index)
            except Exception as exc:
                self._db_available = False
                logger.warning("db_persist_transcript_background_error", call_id=call_id, error=str(exc))

        task = asyncio.create_task(_wrapper())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

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
        if not getattr(self, "_db_available", True) or self._settings.is_testing:
            return

        try:
            async with asyncio.timeout(1.5):
                from app.db.session import session_context
                from app.repositories.demo_call_repository import DemoCallRepository

                async with session_context() as session:
                    repo = DemoCallRepository(session)
                    call_rec = await repo.get_call_record(call_id)
                    if call_rec is None and call_id in self._calls:
                        await repo.save_call(self._calls[call_id])
                    await repo.record_handoff(
                        call_id=call_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        department=department,
                        reason=reason,
                        notes=notes,
                    )
        except Exception as exc:
            self._db_available = False
            logger.warning("db_persist_handoff_failed_fallback_in_memory", call_id=call_id, error=str(exc))

    async def _record_transcript_turn(self, call: DemoCall, turn: TranscriptTurn) -> None:
        """Append turn to in-memory transcript under lock and schedule persistence to database."""

        async with self._lock:
            turn_index = len(call.transcript)

            call.transcript.append(turn)

        self._schedule_persist_transcript_turn(call.call_id, turn, turn_index)

    
    async def load_from_db(self, limit: int = 5000) -> int:
        """Re-hydrate in-memory call cache from PostgreSQL."""

        try:
            async with asyncio.timeout(15.0):
                from app.db.session import session_context
                from app.repositories.demo_call_repository import DemoCallRepository

                async with session_context() as session:
                    repo = DemoCallRepository(session)
                    calls = await repo.list_calls(limit=limit)

                    async with self._lock:
                        for call in calls:
                            if call.call_id not in self._calls:
                                self._calls[call.call_id] = call
                            elif self._calls[call.call_id].status in (CallStatus.COMPLETED, CallStatus.FAILED):
                                self._calls[call.call_id] = call

                    logger.info(
                        "rehydrated_calls_from_db",
                        count=len(calls),
                    )
                    return len(calls)

        except Exception:
            logger.exception("db_load_calls_failed")
            return 0

    async def get_or_load_call(self, call_id: str) -> DemoCall | None:
        """Retrieve a call from cache, or attempt to load from database if missing."""

        call = self.get_call(call_id)

        if call is not None:
            return call

        try:
            async with asyncio.timeout(5.0):
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
            logger.error("db_get_call_failed", call_id=call_id, error=str(exc))

        return None

    async def get_calls_async(
        self,
        status_filter: str | None = None,
        search_query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DemoCall]:
        """Fetch calls from persistent PostgreSQL storage, merging with active in-memory state."""
        try:
            from app.db.session import session_context
            from app.repositories.demo_call_repository import DemoCallRepository

            async with session_context() as session:
                repo = DemoCallRepository(session)
                db_calls = await repo.list_calls(limit=1000)
                async with self._lock:
                    for c in db_calls:
                        if c.call_id not in self._calls:
                            self._calls[c.call_id] = c
                        elif self._calls[c.call_id].status in (CallStatus.COMPLETED, CallStatus.FAILED):
                            self._calls[c.call_id] = c
        except Exception as exc:
            logger.warning("db_get_calls_async_sync_failed", error=str(exc))

        return self.list_calls(
            status_filter=status_filter,
            search_query=search_query,
            limit=limit,
            offset=offset,
        )

    async def get_metrics_async(self) -> CallMetrics:
        """Fetch metrics computed from persisted calls and active sessions."""
        try:
            from app.db.session import session_context
            from app.repositories.demo_call_repository import DemoCallRepository

            async with session_context() as session:
                repo = DemoCallRepository(session)
                db_calls = await repo.list_calls(limit=5000)
                async with self._lock:
                    for c in db_calls:
                        if c.call_id not in self._calls:
                            self._calls[c.call_id] = c
                        elif self._calls[c.call_id].status in (CallStatus.COMPLETED, CallStatus.FAILED):
                            self._calls[c.call_id] = c
        except Exception as exc:
            logger.warning("db_get_metrics_async_sync_failed", error=str(exc))

        return self.get_metrics()

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
        demo_extension: str = "1001",
        language: str | None = None,
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
                extension=demo_extension,
                language=language,
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
                    "extension": demo_extension,
                    "simulated_telephony_channel": f"SIP/LEO-Ext-{demo_extension}-WebRTC-Trunk (Simulated)",
                    "audio_codec": "Opus 48kHz (Simulated)",
                    "region": "ap-south-1",
                    "initial_message": initial_message,
                    "ivr_active": True,
                    "ivr_level": 1,
                    "language_locale": "en-IN",
                    "demo_label": "DEMO / SIMULATED CALL",
                },
            )

            self._calls[call_id] = call

        # Persist to database asynchronously
        self._schedule_persist_call(call)

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
        demo_extension: str = "1001",
        language: str = "auto",
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

            # Resolve language metadata if specific language selected
            selected_lang_name = None if language == "auto" else language
            selected_lang_code = None
            if selected_lang_name:
                for l_item in SUPPORTED_DEMO_LANGUAGES:
                    if l_item["name"].lower() == selected_lang_name.lower() or l_item["code"].lower() == selected_lang_name.lower():
                        selected_lang_name = l_item["name"]
                        selected_lang_code = l_item["code"]
                        break

            call = DemoCall(
                call_id=call_id,
                session_id=session_id,
                caller_id=f"CR-{random.randint(1000, 9999)}",
                caller_name=customer_name,
                caller_phone_masked=masked_phone,
                extension=demo_extension,
                language=selected_lang_name,
                language_code=selected_lang_code,
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
                    "extension": demo_extension,
                    "call_purpose": call_purpose,
                    "selected_language": selected_lang_name,
                    "language_locale": selected_lang_code or "en-IN",
                    "simulated_telephony_channel": f"SIP/LEO-Ext-{demo_extension}-Outbound-Trunk (Simulated)",
                    "audio_codec": "Opus 48kHz (Simulated)",
                    "region": "ap-south-1",
                    "demo_label": "DEMO / SIMULATED CALL",
                    "initial_message": f"Hello, I understand you are calling regarding {call_purpose}.",
                },
            )

            self._calls[call_id] = call

        # Persist to database asynchronously
        self._schedule_persist_call(call)

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

        # Persist updated status & duration
        if target_status in (CallStatus.COMPLETED, CallStatus.FAILED):
            try:
                await self._persist_call(call)
            except Exception as exc:
                logger.error("db_persist_terminal_call_failed", call_id=call_id, error=str(exc))
        else:
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
                text="Call connected via LEO AI Virtual Agent Gateway (Global Projects Voice Bridge).",
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
                purpose = call.metadata.get("call_purpose", "your project inquiry")
                lang = call.metadata.get("selected_language", "auto")
                ext_str = getattr(call, "extension", "1001") or "1001"
                if lang and ("hi" in lang.lower() or "hindi" in lang.lower()):
                    greeting_text = (
                        f"Namaste! Main LEO bol raha hoon Global Projects se (Extension {ext_str}) "
                        f"{purpose} ke sambandh mein. Kya yeh aapse baat karne ka sahi samay hai?"
                    )
                else:
                    greeting_text = (
                        f"Hello, this is LEO calling from Global Projects (Extension {ext_str}) "
                        f"regarding {purpose}. Is this a convenient time to speak?"
                    )
            else:
                # English welcome greeting for incoming calls to demo extension 1001
                call.metadata["ivr_active"] = True
                call.metadata["ivr_level"] = 1
                call.metadata["language_locale"] = "en-IN"

            if call.direction == CallDirection.INBOUND:
                greeting_text = (
                    "Hello! Welcome to LEO, your AI voice assistant. "
                    "I'm here to help you."
                )
            # (outbound greeting_text already set above)

            greeting_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.LEO,
                text=greeting_text,
                timestamp=datetime.now(UTC),
                metadata={
                    "turn_type": "greeting",
                    "language_locale": "en-IN",
                    "ivr_active": call.metadata.get("ivr_active", False),
                    "extension": getattr(call, "extension", "1001"),
                },
            )

            await self._record_transcript_turn(call, greeting_turn)

            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": greeting_turn.model_dump(mode="json"),
                },
            )

            # Emit IVR language-selection instructions for inbound calls
            if call.direction == CallDirection.INBOUND:
                await asyncio.sleep(0.1)
                ivr_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.LEO,
                    text=(
                        "Please select your preferred language. "
                        "For English, press 1. "
                        "For Hindi, press 2. "
                        "For other Indian languages, press 3."
                    ),
                    timestamp=datetime.now(UTC),
                    metadata={
                        "turn_type": "ivr_instructions",
                        "language_locale": "en-IN",
                        "ivr_active": True,
                    },
                )
                await self._record_transcript_turn(call, ivr_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {
                        "call_id": call_id,
                        "turn": ivr_turn.model_dump(mode="json"),
                    },
                )

            # Step 3: Transition to ACTIVE listening state

            await self.transition_state(call_id, CallStatus.ACTIVE)

            # Automatically dispatch initial customer inquiry after short pause if not waiting in IVR
            initial_msg = call.metadata.get("initial_message")

            if initial_msg and not call.metadata.get("ivr_active"):
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

        elif action == CallActionType.CLEAR:
            return await self.clear_transcript(call_id)

        elif action == CallActionType.DTMF:
            return await self.handle_dtmf(call_id, digit=reason or "1")

        elif action == CallActionType.END:
            await self.transition_state(call_id, CallStatus.CALL_ENDING)

            await self.transition_state(call_id, CallStatus.COMPLETED)

        return call

    async def handle_dtmf(self, call_id: str, digit: str) -> DemoCall:
        """Process DTMF keypad digit for active call session."""
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        digit_str = str(digit).strip()
        call.metadata["last_dtmf"] = digit_str

        ivr_active = call.metadata.get("ivr_active", False)
        ivr_level = int(call.metadata.get("ivr_level", 1) or 1)

        # If IVR is not active (e.g. normal conversation phase), record DTMF signaling event
        if not ivr_active:
            sys_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.SYSTEM,
                text=f"Customer pressed {digit_str}",
                timestamp=datetime.now(UTC),
                metadata={"turn_type": "dtmf_digit", "digit": digit_str},
            )
            await self._record_transcript_turn(call, sys_turn)
            await self.broadcast_event(
                "transcript_turn",
                {"call_id": call_id, "turn": sys_turn.model_dump(mode="json")},
            )
            await self.broadcast_event("dtmf_received", {"call_id": call_id, "digit": digit_str})
            return call

        # Level 1: Main IVR Menu (1: English, 2: Hindi, 3: Other Indian Languages)
        if ivr_level == 1:
            if digit_str == "1":
                return await self.select_language(call_id, language_code="en-IN", language_name="English", key="1")
            elif digit_str == "2":
                return await self.select_language(call_id, language_code="hi-IN", language_name="Hindi", key="2")
            elif digit_str == "3":
                sys_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.SYSTEM,
                    text="Customer pressed 3 (Other Indian Languages)",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "dtmf_digit", "digit": "3", "menu": "main"},
                )
                await self._record_transcript_turn(call, sys_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": sys_turn.model_dump(mode="json")},
                )

                async with self._lock:
                    call.metadata["ivr_level"] = 2

                submenu_prompt = (
                    "Please select your preferred language. "
                    "For Tamil, press 1. "
                    "Telugu, press 2. "
                    "Kannada, press 3. "
                    "Malayalam, press 4. "
                    "Tulu, press 5. "
                    "Konkani, press 6. "
                    "Urdu, press 7."
                )
                submenu_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.LEO,
                    text=submenu_prompt,
                    timestamp=datetime.now(UTC),
                    metadata={
                        "turn_type": "ivr_submenu_instructions",
                        "language_locale": "en-IN",
                        "ivr_active": True,
                        "ivr_level": 2,
                    },
                )
                await self._record_transcript_turn(call, submenu_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": submenu_turn.model_dump(mode="json")},
                )
                self._schedule_persist_call(call)
                await self.broadcast_event(
                    "call_state_changed",
                    {
                        "call_id": call_id,
                        "previous_status": call.status,
                        "new_status": call.status,
                        "call": call.model_dump(mode="json"),
                    },
                )
                return call
            elif digit_str in ("*", "0", "#"):
                sys_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.SYSTEM,
                    text=f"Customer pressed {digit_str}",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "dtmf_digit", "digit": digit_str},
                )
                await self._record_transcript_turn(call, sys_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": sys_turn.model_dump(mode="json")},
                )
                repeat_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.LEO,
                    text="For English, press 1. For Hindi, press 2. For other Indian languages, press 3.",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "ivr_instructions", "language_locale": "en-IN", "ivr_active": True},
                )
                await self._record_transcript_turn(call, repeat_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": repeat_turn.model_dump(mode="json")},
                )
                return call
            else:
                sys_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.SYSTEM,
                    text=f"Customer pressed {digit_str}",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "dtmf_digit", "digit": digit_str},
                )
                await self._record_transcript_turn(call, sys_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": sys_turn.model_dump(mode="json")},
                )
                inv_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.LEO,
                    text="Invalid selection. For English, press 1. For Hindi, press 2. For other Indian languages, press 3.",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "ivr_instructions", "language_locale": "en-IN", "ivr_active": True},
                )
                await self._record_transcript_turn(call, inv_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": inv_turn.model_dump(mode="json")},
                )
                return call

        # Level 2: Submenu (1: Tamil, 2: Telugu, 3: Kannada, 4: Malayalam, 5: Tulu, 6: Konkani, 7: Urdu)
        elif ivr_level == 2:
            submenu_map = {
                "1": ("ta-IN", "Tamil"),
                "2": ("te-IN", "Telugu"),
                "3": ("kn-IN", "Kannada"),
                "4": ("ml-IN", "Malayalam"),
                "5": ("tcy-IN", "Tulu"),
                "6": ("kok-IN", "Konkani"),
                "7": ("ur-IN", "Urdu"),
            }
            if digit_str in submenu_map:
                code, name = submenu_map[digit_str]
                return await self.select_language(call_id, language_code=code, language_name=name, key=digit_str)
            elif digit_str in ("*", "0", "#"):
                sys_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.SYSTEM,
                    text=f"Customer pressed {digit_str} (Return to Main Menu)",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "dtmf_digit", "digit": digit_str},
                )
                await self._record_transcript_turn(call, sys_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": sys_turn.model_dump(mode="json")},
                )
                async with self._lock:
                    call.metadata["ivr_level"] = 1
                return_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.LEO,
                    text="Returning to main menu. For English, press 1. For Hindi, press 2. For other Indian languages, press 3.",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "ivr_instructions", "language_locale": "en-IN", "ivr_active": True, "ivr_level": 1},
                )
                await self._record_transcript_turn(call, return_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": return_turn.model_dump(mode="json")},
                )
                self._schedule_persist_call(call)
                await self.broadcast_event(
                    "call_state_changed",
                    {
                        "call_id": call_id,
                        "previous_status": call.status,
                        "new_status": call.status,
                        "call": call.model_dump(mode="json"),
                    },
                )
                return call
            else:
                sys_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.SYSTEM,
                    text=f"Customer pressed {digit_str}",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "dtmf_digit", "digit": digit_str},
                )
                await self._record_transcript_turn(call, sys_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": sys_turn.model_dump(mode="json")},
                )
                inv_turn = TranscriptTurn(
                    id=f"turn_{uuid.uuid4().hex[:8]}",
                    role=SpeakerRole.LEO,
                    text="Invalid selection. For Tamil, press 1. Telugu, press 2. Kannada, press 3. Malayalam, press 4. Tulu, press 5. Konkani, press 6. Urdu, press 7.",
                    timestamp=datetime.now(UTC),
                    metadata={"turn_type": "ivr_submenu_instructions", "language_locale": "en-IN", "ivr_active": True, "ivr_level": 2},
                )
                await self._record_transcript_turn(call, inv_turn)
                await self.broadcast_event(
                    "transcript_turn",
                    {"call_id": call_id, "turn": inv_turn.model_dump(mode="json")},
                )
                return call

        return call

    async def select_language(
        self,
        call_id: str,
        language_code: str,
        language_name: str | None = None,
        key: str | None = None,
    ) -> DemoCall:
        """Register IVR language selection and welcome caller in their selected language."""
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        # Find matching language config
        lang_item = next((item for item in SUPPORTED_DEMO_LANGUAGES if item["code"].lower() == language_code.lower()), None)
        if not lang_item and language_name:
            lang_item = next((item for item in SUPPORTED_DEMO_LANGUAGES if item["name"].lower() == language_name.lower()), None)

        display_name = lang_item["name"] if lang_item else (language_name or language_code)
        native_name = lang_item["native_name"] if lang_item else display_name
        greeting = lang_item["greeting"] if lang_item else f"Welcome! You have selected {display_name}."

        key_label = f"Keypad {key} Pressed: " if key else "Language Selected: "
        customer_selection_turn = TranscriptTurn(
            id=f"turn_{uuid.uuid4().hex[:8]}",
            role=SpeakerRole.SYSTEM,
            text=f"Customer pressed {key} ({native_name} - {display_name})" if key else f"Customer selected {native_name} ({display_name})",
            timestamp=datetime.now(UTC),
            metadata={"turn_type": "language_selection", "language_code": language_code, "key": key},
        )
        await self._record_transcript_turn(call, customer_selection_turn)
        await self.broadcast_event(
            "transcript_turn",
            {"call_id": call_id, "turn": customer_selection_turn.model_dump(mode="json")},
        )

        async with self._lock:
            call.language = display_name
            call.language_code = language_code
            call.metadata["selected_language"] = display_name
            call.metadata["language_locale"] = language_code
            call.metadata["language_native"] = native_name
            call.metadata["ivr_active"] = False
            call.metadata["ivr_level"] = 0

        # LEO confirmation welcome greeting turn in the selected language:
        leo_welcome_turn = TranscriptTurn(
            id=f"turn_{uuid.uuid4().hex[:8]}",
            role=SpeakerRole.LEO,
            text=greeting,
            timestamp=datetime.now(UTC),
            metadata={
                "turn_type": "language_welcome",
                "language_locale": language_code,
                "detected_language": language_code.split("-")[0],
                "sample_inquiries": lang_item.get("sample_inquiries", []) if lang_item else [],
            },
        )
        await self._record_transcript_turn(call, leo_welcome_turn)
        await self.broadcast_event(
            "transcript_turn",
            {"call_id": call_id, "turn": leo_welcome_turn.model_dump(mode="json")},
        )

        self._schedule_persist_call(call)
        await self.broadcast_event(
            "call_state_changed",
            {
                "call_id": call_id,
                "previous_status": call.status,
                "new_status": call.status,
                "call": call.model_dump(mode="json"),
            },
        )
        return call

    async def clear_transcript(self, call_id: str) -> DemoCall:
        """Clear all turns from the specified call transcript."""
        call = self.get_call(call_id)
        if not call:
            raise KeyError(f"Call {call_id} not found")

        async with self._lock:
            call.transcript = []

        self._schedule_persist_call(call)
        await self.broadcast_event(
            "call_state_changed",
            {
                "call_id": call_id,
                "previous_status": call.status,
                "new_status": call.status,
                "call": call.model_dump(mode="json"),
            },
        )
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

    def _get_localized_response(self, query: str, lang_code: str, default_text: str) -> str:
        lang = (lang_code or "").lower()
        q = (query or "").lower()
        if any(h in q for h in ["human", "agent", "operator", "specialist", "इंसान", "बात", "உதவி", "మనిషి", "ಅಧಿಕಾರಿ"]):
            if "hi" in lang:
                return "जी बिल्कुल, मैं आपकी कॉल हमारे वरिष्ठ कस्टमर सपोर्ट विशेषज्ञ को ट्रांसफर कर रहा हूँ। कृपया लाइन पर बने रहें।"
            elif "ta" in lang:
                return "நிச்சயமாக, உங்கள் அழைப்பை எங்கள் வாடிக்கையாளர் சேவை நிபுணருக்கு மாற்றுகிறேன். தயவுசெய்து காத்திருக்கவும்."
            elif "te" in lang:
                return "తప్పకుండా, మీ కాల్‌ను మా కస్టమర్ కేర్ స్పెషలిస్ట్‌కు బదిలీ చేస్తున్నాను. దయచేసి లైన్‌లో వేచి ఉండండి."
            elif "kn" in lang:
                return "ಖಂಡಿತ, ನಿಮ್ಮ ಕರೆಯನ್ನು ನಮ್ಮ ತಜ್ಞ ಅಧಿಕಾರಿಗೆ ವರ್ಗಾಯಿಸುತ್ತಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಲೈನ್‌ನಲ್ಲಿ ನಿರೀಕ್ಷಿಸಿ."
            elif "ml" in lang:
                return "തീർച്ചയായും, നിങ്ങളുടെ കോൾ ഞങ്ങളുടെ കസ്റ്റമർ സപ്പോർട്ട് സ്പെഷ്യലിസ്റ്റിലേക്ക് മാറ്റുന്നു. ദയവായി കാത്തിരിക്കുക."
            elif "ur" in lang:
                return "جی بالکل، میں آپ کی کال کسٹمر کیئر نمائندے کو ٹرانسفر کر رہا ہوں۔ براہ کرم لائن پر رہیں۔"
            else:
                return "Certainly, I am transferring your call to a human specialist right away. Please hold the line."

        if "hi" in lang:
            return "ग्लोबल प्रोजेक्ट्स एंटरप्राइज क्लाउड आर्किटेक्चर, आईटी इंफ्रास्ट्रक्चर और डिजिटल ट्रांसफॉर्मेशन समाधान प्रदान करता है। मैं आपकी परियोजना में कैसे सहायता कर सकता हूँ?"
        elif "ta" in lang:
            return "குளோபல் ப்ராஜெக்ட்ஸ் நிறுவனம் கிளவுட் மேலாண்மை, தரவு பாதுகாப்பு மற்றும் மென்பொருள் மேம்பாட்டு சேவைகளை வழங்குகிறது. உங்களுக்கு எவ்வாறு உதவ வேண்டும்?"
        elif "te" in lang:
            return "గ్లోబల్ ప్రాజెక్ట్స్ సంస్థ ఎంటర్‌ప్రైజ్ క్లౌడ్, డేటా సెక్యూరిటీ మరియు సాఫ్ట్‌వేర్ అభివృద్ధి సేవలను అందిస్తుంది. మీకు ఏ వివరాలు కావాలి?"
        elif "kn" in lang:
            return "ಗ್ಲೋಬಲ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಎಂಟರ್‌ಪ್ರೈಸ್ ಕ್ಲೌಡ್ ಸಲ್ಯೂಷನ್ಸ್ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಸೇವೆಗಳನ್ನು ಒದಗಿಸುತ್ತದೆ. ನಿಮ್ಮ ಯೋಜನೆಗೆ ನಾವು ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
        elif "ml" in lang:
            return "ഗ്ലോബൽ പ്രോജക്ട്സ് എന്റർപ്രൈസ് ക്ലൗഡ് കൺസൾട്ടിംഗ്, ഡിജിറ്റൽ പരിവർത്തനം എന്നീ സേവനങ്ങൾ നൽകുന്നു. നിങ്ങൾക്ക് ഏത് വിവരമാണ് ആവശ്യം?"
        elif "tcy" in lang:
            return "ಗ್ಲೋಬಲ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಸಂಸ್ಥೆ ಕ್ಲೌಡ್ ಸೇವೆಲು ಬೊಕ್ಕ ಐಟಿ ಬೆಂಬಲ ಕೊರ್ಪುಂಡು. ಈರೆನ ಪ್ರಾಜೆಕ್ಟ್‌ಗ್ ನಮ ಎಂಚ ಸಾಯ ಮಲ್ಪೊಲಿ?"
        elif "kok" in lang:
            return "ग्लोबल प्रोजेक्ट्स कंपनी क्लाउड आनी सॉफ्टवेअर सोल्यूशन्स दिता. तुमकां कसली म्हायती जाय?"
        elif "ur" in lang:
            return "گلوبل پروجیکٹس انٹرپرائز کلاؤڈ آرکیٹیکچر اور آئی ٹی سروسز فراہم کرتا ہے۔ ہم آپ کے پروجیکٹ میں کس طرح مدد کر سکتے ہیں؟"
        elif "en" in lang:
            return "Global Projects provides enterprise cloud architecture, IT infrastructure management, and secure software development services. How can I assist with your project requirements today?"
        return default_text

    async def send_customer_message(
        self,
        call_id: str,
        message_text: str,
        request_id: str | None = None,
    ) -> TranscriptTurn:
        """Customer speaks a turn, triggering LEO AI response processing."""

        t_turn_start = time.perf_counter()
        req_id = request_id or str(uuid.uuid4())

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
                text="Of course. I can't connect calls directly, but you can reach the Global Projects team through their official channels for a direct conversation.",
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
        logger.info(
            "stage_timing",
            request_id=req_id,
            call_id=call_id,
            stage="language_detection",
            stage_name="Language detection",
            duration_ms=round((time.perf_counter() - t_turn_start) * 1000.0, 2),
        )

        # Create TurnManager agent turn token
        agent_task = asyncio.current_task()
        agent_token = await self.turn_manager.create_agent_turn(call.session_id, task=agent_task)

        # 4. Invoke the real LEO RAG Orchestrator
        try:
            # Build conversation history from recent turns
            history_messages: list[Message] = []
            for t in call.transcript[-8:-1]:
                if t.role == SpeakerRole.CUSTOMER:
                    history_messages.append(Message(role="user", content=t.text))
                elif t.role == SpeakerRole.LEO:
                    history_messages.append(Message(role="assistant", content=t.text))

            logger.info(
                "stage_timing",
                request_id=req_id,
                call_id=call_id,
                stage="conversation_history_loaded",
                stage_name="Conversation history loaded",
                duration_ms=round((time.perf_counter() - t_turn_start) * 1000.0, 2),
            )

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
                cancellation_token=agent_token,
                request_id=req_id,
                call_id=call_id,
            )
            if not await self.turn_manager.is_current(call.session_id, agent_token):
                raise asyncio.CancelledError()
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

            final_response_text = ai_response.response_text
            effective_lang = lang_locale or call.metadata.get("language_locale", "en-IN")
            if (
                getattr(self._llm, "provider_name", "") == "mock"
                or "[Mock LEO response]" in final_response_text
            ):
                final_response_text = self._get_localized_response(message_text, effective_lang, final_response_text)

            leo_turn = TranscriptTurn(
                id=f"turn_{uuid.uuid4().hex[:8]}",
                role=SpeakerRole.LEO,
                text=final_response_text,
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
                    "language_locale": effective_lang,
                },
            )

            # Publish only while this token still owns the session.
            if not await self.turn_manager.is_current(call.session_id, agent_token):
                raise asyncio.CancelledError()
            # 5. Transition to SPEAKING state (acoustic / speech visualization)
            await self.transition_state(call_id, CallStatus.SPEAKING)
            if not await self.turn_manager.is_current(call.session_id, agent_token):
                raise asyncio.CancelledError()
            await self._record_transcript_turn(call, leo_turn)

            await self.broadcast_event(
                "transcript_turn",
                {
                    "call_id": call_id,
                    "turn": leo_turn.model_dump(mode="json"),
                },
            )

            # Trigger non-blocking TTS pre-warming & stage timing
            asyncio.create_task(
                self._prewarm_tts_turn(
                    call_id=call_id,
                    turn_id=leo_turn.id,
                    text=leo_turn.text,
                    language_code=lang_locale,
                    request_id=req_id,
                    turn_start_time=t_turn_start,
                )
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
            await self.turn_manager.unregister_turn_task(call.session_id, agent_task)

        except asyncio.CancelledError:
            logger.info("customer_turn_cancelled", call_id=call_id, token_id=agent_token.token_id)
            current_task = asyncio.current_task()
            if current_task and hasattr(current_task, "uncancel"):
                current_task.uncancel()

            await self.turn_manager.complete_agent_turn(call.session_id, agent_token)
            await self.turn_manager.unregister_turn_task(call.session_id, agent_task)

            try:
                await self.transition_state(call_id, CallStatus.ACTIVE)
            except Exception:
                pass

            return customer_turn

        except Exception as exc:
            logger.error("leo_brain_invocation_failed", call_id=call_id, error=str(exc))

            # Mark TurnManager complete

            await self.turn_manager.complete_agent_turn(call.session_id, agent_token)
            await self.turn_manager.unregister_turn_task(call.session_id, agent_task)
            await self.turn_manager.unregister_turn_task(call.session_id, agent_task)

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

        from app.voice.speech_normalizer import normalize_text_for_speech

        spoken_text = normalize_text_for_speech(text, language_code=language_code)
        req = SpeechSynthesisRequest(
            text=spoken_text or text,
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

    async def _prewarm_tts_turn(
        self,
        call_id: str,
        turn_id: str,
        text: str,
        language_code: str | None,
        request_id: str,
        turn_start_time: float,
    ) -> None:
        """Asynchronously pre-warm TTS for this turn and cache generated frames."""
        if self._tts is None or not text.strip():
            return
        t_tts_start = time.perf_counter()
        dur_start = (t_tts_start - turn_start_time) * 1000.0
        logger.info(
            "stage_timing",
            request_id=request_id,
            call_id=call_id,
            stage="tts_started",
            stage_name="TTS started",
            duration_ms=round(dur_start, 2),
        )
        frames: list[bytes] = []
        first_chunk_logged = False
        try:
            async for frame in self.stream_turn_audio_framed(
                text=text,
                turn_id=None,
                language_code=language_code,
            ):
                if not first_chunk_logged:
                    first_chunk_logged = True
                    audio_dur = (time.perf_counter() - turn_start_time) * 1000.0
                    logger.info(
                        "stage_timing",
                        request_id=request_id,
                        call_id=call_id,
                        stage="tts_first_audio_chunk",
                        stage_name="First audio chunk generated",
                        duration_ms=round(audio_dur, 2),
                    )
                frames.append(frame)
            if len(self._audio_cache) > 50:
                self._audio_cache.pop(next(iter(self._audio_cache)))
            self._audio_cache[turn_id] = frames
        except Exception as exc:
            logger.debug("prewarm_tts_turn_skipped", turn_id=turn_id, error=str(exc))

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

        # Serve from in-memory pre-warmed audio cache if available
        if turn_id and turn_id in self._audio_cache:
            for frame in self._audio_cache[turn_id]:
                if cancellation_token and getattr(cancellation_token, "is_cancelled", False):
                    break
                yield frame
            return

        from app.voice.speech_normalizer import normalize_text_for_speech

        spoken_text = normalize_text_for_speech(text, language_code=language_code)
        req = SpeechSynthesisRequest(
            text=spoken_text or text,
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

        # Serve from in-memory pre-warmed audio cache if available
        if turn_id and turn_id in self._audio_cache:
            for frame in self._audio_cache[turn_id]:
                if cancellation_token and getattr(cancellation_token, "is_cancelled", False):
                    break
                yield frame
            return

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
                    if cancellation_token is not None:
                        marked = await self.turn_manager.mark_speaking(
                            call.session_id, cancellation_token
                        )
                        if not marked:
                            return
                    try:
                        await self.transition_state(call_id, CallStatus.SPEAKING)
                    except Exception:
                        pass

                # Yield text frame so client can show transcript incrementally

                yield pack_text_chunk(sentence)

                # Synthesize TTS for this sentence

                if stream_method is not None:
                    from app.voice.speech_normalizer import normalize_text_for_speech

                    spoken_sentence = normalize_text_for_speech(sentence, language_code=req.language_code)
                    req_s = SpeechSynthesisRequest(
                        text=spoken_sentence or sentence,
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

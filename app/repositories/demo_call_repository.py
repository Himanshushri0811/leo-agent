"""
Repository for DemoCallRecord, DemoCallTranscriptRecord, and DemoAgentHandoffRecord entities.

Encapsulates database operations for call persistence, transcript logging,
and escalation history.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.demo import (
    DemoAgentHandoffRecord,
    DemoCallRecord,
    DemoCallTranscriptRecord,
)
from app.schemas.demo import (
    CallDirection,
    CallStatus,
    DemoCall,
    EscalationReason,
    SpeakerRole,
    TranscriptTurn,
)


class DemoCallRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _record_to_schema(record: DemoCallRecord) -> DemoCall:
        """Converts an ORM DemoCallRecord into a Pydantic DemoCall schema."""
        transcript_turns = [
            TranscriptTurn(
                id=t.id,
                role=SpeakerRole(t.role),
                text=t.text,
                timestamp=t.timestamp,
                sources=t.sources or [],
                metadata=t.turn_metadata or {},
                confidence_score=t.confidence_score,
            )
            for t in (record.transcripts or [])
        ]

        escalation_reason = (
            EscalationReason(record.escalation_reason)
            if record.escalation_reason and record.escalation_reason in EscalationReason._value2member_map_
            else None
        )

        meta = record.call_metadata or {}
        raw_direction = meta.get("direction", "inbound")
        direction = (
            CallDirection(raw_direction)
            if raw_direction in CallDirection._value2member_map_
            else CallDirection.INBOUND
        )

        return DemoCall(
            call_id=record.id,
            session_id=record.session_id,
            caller_id=record.caller_id,
            caller_name=record.caller_name,
            caller_phone_masked=record.caller_phone_masked,
            status=CallStatus(record.status),
            is_muted=record.is_muted,
            is_simulated=record.is_simulated,
            auto_answer_delay_sec=record.auto_answer_delay_sec,
            created_at=record.created_at,
            connected_at=record.connected_at,
            ended_at=record.ended_at,
            duration_seconds=record.duration_seconds,
            escalated=record.escalated,
            escalation_reason=escalation_reason,
            escalation_notes=record.escalation_notes,
            assigned_agent_id=record.assigned_agent_id,
            assigned_agent_name=record.assigned_agent_name,
            transcript=transcript_turns,
            latest_error=record.latest_error,
            direction=direction,
            turn_state=meta.get("turn_state", "idle"),
            metadata=meta,
        )

    async def get_call_record(self, call_id: str) -> DemoCallRecord | None:
        """Fetch the raw ORM record for a call."""
        stmt = (
            select(DemoCallRecord)
            .where(DemoCallRecord.id == call_id)
            .options(
                selectinload(DemoCallRecord.transcripts),
                selectinload(DemoCallRecord.handoffs),
            )
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_call(self, call_id: str) -> DemoCall | None:
        """Fetch a call converted to a DemoCall schema or None if not found."""
        record = await self.get_call_record(call_id)
        if record is None:
            return None
        return self._record_to_schema(record)

    async def list_calls(self, limit: int = 50, offset: int = 0) -> list[DemoCall]:
        """List persisted demo calls ordered by creation time descending."""
        stmt = (
            select(DemoCallRecord)
            .options(
                selectinload(DemoCallRecord.transcripts),
                selectinload(DemoCallRecord.handoffs),
            )
            .order_by(desc(DemoCallRecord.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        records = result.scalars().all()
        return [self._record_to_schema(r) for r in records]

    async def save_call(self, call: DemoCall) -> DemoCallRecord:
        """Create or update a DemoCall record in the database."""
        record = await self.get_call_record(call.call_id)
        meta = dict(call.metadata or {})
        meta["direction"] = str(call.direction)
        meta["turn_state"] = str(call.turn_state)

        if record is None:
            record = DemoCallRecord(
                id=call.call_id,
                session_id=call.session_id,
                caller_id=call.caller_id,
                caller_name=call.caller_name,
                caller_phone_masked=call.caller_phone_masked,
                status=str(call.status),
                is_muted=call.is_muted,
                is_simulated=call.is_simulated,
                auto_answer_delay_sec=call.auto_answer_delay_sec,
                created_at=call.created_at,
                connected_at=call.connected_at,
                ended_at=call.ended_at,
                duration_seconds=call.duration_seconds,
                escalated=call.escalated,
                escalation_reason=str(call.escalation_reason) if call.escalation_reason else None,
                escalation_notes=call.escalation_notes,
                assigned_agent_id=call.assigned_agent_id,
                assigned_agent_name=call.assigned_agent_name,
                latest_error=call.latest_error,
                call_metadata=meta,
            )
            self._session.add(record)
        else:
            record.session_id = call.session_id
            record.caller_name = call.caller_name
            record.caller_phone_masked = call.caller_phone_masked
            record.status = str(call.status)
            record.is_muted = call.is_muted
            record.is_simulated = call.is_simulated
            record.auto_answer_delay_sec = call.auto_answer_delay_sec
            record.connected_at = call.connected_at
            record.ended_at = call.ended_at
            record.duration_seconds = call.duration_seconds
            record.escalated = call.escalated
            record.escalation_reason = str(call.escalation_reason) if call.escalation_reason else None
            record.escalation_notes = call.escalation_notes
            record.assigned_agent_id = call.assigned_agent_id
            record.assigned_agent_name = call.assigned_agent_name
            record.latest_error = call.latest_error
            record.call_metadata = meta

        await self._session.flush()
        return record

    async def add_transcript_turn(
        self,
        call_id: str,
        turn: TranscriptTurn,
        turn_index: int,
    ) -> DemoCallTranscriptRecord:
        """Persist a single transcript message for a call."""
        # Ensure parent call exists or is flushed
        stmt = select(DemoCallTranscriptRecord).where(DemoCallTranscriptRecord.id == turn.id)
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing:
            return existing

        record = DemoCallTranscriptRecord(
            id=turn.id,
            call_id=call_id,
            turn_index=turn_index,
            role=str(turn.role),
            text=turn.text,
            timestamp=turn.timestamp,
            confidence_score=turn.confidence_score,
            sources=turn.sources or [],
            turn_metadata=turn.metadata or {},
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def record_handoff(
        self,
        call_id: str,
        agent_id: str,
        agent_name: str,
        department: str,
        reason: str | None = None,
        notes: str | None = None,
    ) -> DemoAgentHandoffRecord:
        """Persist an agent escalation handoff record."""
        handoff = DemoAgentHandoffRecord(
            call_id=call_id,
            agent_id=agent_id,
            agent_name=agent_name,
            department=department,
            reason=reason,
            notes=notes,
            created_at=datetime.now(UTC),
        )
        self._session.add(handoff)
        await self._session.flush()
        return handoff

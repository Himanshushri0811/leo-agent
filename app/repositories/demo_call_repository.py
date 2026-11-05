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
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
        meta = dict(call.metadata or {})
        meta["direction"] = str(call.direction)
        meta["turn_state"] = str(call.turn_state)

        insert_values = {
            "id": call.call_id,
            "session_id": call.session_id,
            "caller_id": call.caller_id,
            "caller_name": call.caller_name,
            "caller_phone_masked": call.caller_phone_masked,
            "status": str(call.status),
            "is_muted": call.is_muted,
            "is_simulated": call.is_simulated,
            "auto_answer_delay_sec": call.auto_answer_delay_sec,
            "created_at": call.created_at,
            "connected_at": call.connected_at,
            "ended_at": call.ended_at,
            "duration_seconds": call.duration_seconds,
            "escalated": call.escalated,
            "escalation_reason": str(call.escalation_reason) if call.escalation_reason else None,
            "escalation_notes": call.escalation_notes,
            "assigned_agent_id": call.assigned_agent_id,
            "assigned_agent_name": call.assigned_agent_name,
            "latest_error": call.latest_error,
            "metadata": meta,
        }

        table = DemoCallRecord.__table__
        stmt = pg_insert(table).values(**insert_values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.id],
            set_={
                "session_id": stmt.excluded.session_id,
                "caller_name": stmt.excluded.caller_name,
                "caller_phone_masked": stmt.excluded.caller_phone_masked,
                "status": stmt.excluded.status,
                "is_muted": stmt.excluded.is_muted,
                "is_simulated": stmt.excluded.is_simulated,
                "auto_answer_delay_sec": stmt.excluded.auto_answer_delay_sec,
                "connected_at": stmt.excluded.connected_at,
                "ended_at": stmt.excluded.ended_at,
                "duration_seconds": stmt.excluded.duration_seconds,
                "escalated": stmt.excluded.escalated,
                "escalation_reason": stmt.excluded.escalation_reason,
                "escalation_notes": stmt.excluded.escalation_notes,
                "assigned_agent_id": stmt.excluded.assigned_agent_id,
                "assigned_agent_name": stmt.excluded.assigned_agent_name,
                "latest_error": stmt.excluded.latest_error,
                "metadata": stmt.excluded.metadata,
            },
        )
        await self._session.execute(stmt)
        await self._session.flush()

        if call.transcript:
            t_table = DemoCallTranscriptRecord.__table__
            for idx, turn in enumerate(call.transcript):
                t_stmt = pg_insert(t_table).values(
                    id=turn.id,
                    call_id=call.call_id,
                    turn_index=idx,
                    role=str(turn.role),
                    text=turn.text,
                    timestamp=turn.timestamp,
                    confidence_score=turn.confidence_score,
                    sources=turn.sources or [],
                    metadata=turn.metadata or {},
                ).on_conflict_do_nothing(index_elements=[t_table.c.id])
                await self._session.execute(t_stmt)
            await self._session.flush()

        record = await self.get_call_record(call.call_id)
        return record  # type: ignore[return-value]

    async def add_transcript_turn(
        self,
        call_id: str,
        turn: TranscriptTurn,
        turn_index: int,
    ) -> DemoCallTranscriptRecord:
        """Persist a single transcript message for a call."""
        t_table = DemoCallTranscriptRecord.__table__
        t_stmt = pg_insert(t_table).values(
            id=turn.id,
            call_id=call_id,
            turn_index=turn_index,
            role=str(turn.role),
            text=turn.text,
            timestamp=turn.timestamp,
            confidence_score=turn.confidence_score,
            sources=turn.sources or [],
            metadata=turn.metadata or {},
        ).on_conflict_do_nothing(index_elements=[t_table.c.id])
        await self._session.execute(t_stmt)
        await self._session.flush()

        stmt = select(DemoCallTranscriptRecord).where(DemoCallTranscriptRecord.id == turn.id)
        return (await self._session.execute(stmt)).scalar_one()

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

"""
ORM models for demo call center persistence.

DemoCallRecord          - persistent record for a demo call session.
DemoCallTranscriptRecord- individual turns within a call transcript.
DemoAgentHandoffRecord  - escalation and human agent assignment events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DemoCallRecord(Base):
    """Stores demo call records and lifecycle metadata."""

    __tablename__ = "demo_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    caller_id: Mapped[str] = mapped_column(String(255), nullable=False)
    caller_name: Mapped[str] = mapped_column(String(255), nullable=False)
    caller_phone_masked: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="ringing", index=True)
    is_muted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    auto_answer_delay_sec: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    escalation_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    escalation_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    assigned_agent_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    assigned_agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latest_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    call_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    transcripts: Mapped[list[DemoCallTranscriptRecord]] = relationship(
        "DemoCallTranscriptRecord",
        back_populates="call",
        order_by="DemoCallTranscriptRecord.turn_index",
        cascade="all, delete-orphan",
    )

    handoffs: Mapped[list[DemoAgentHandoffRecord]] = relationship(
        "DemoAgentHandoffRecord",
        back_populates="call",
        order_by="DemoAgentHandoffRecord.created_at",
        cascade="all, delete-orphan",
    )


class DemoCallTranscriptRecord(Base):
    """Stores individual transcript turns associated with a demo call."""

    __tablename__ = "demo_call_transcripts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    call_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("demo_calls.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sources: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    turn_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    call: Mapped[DemoCallRecord] = relationship("DemoCallRecord", back_populates="transcripts")


class DemoAgentHandoffRecord(Base):
    """Stores human agent escalation / transfer records."""

    __tablename__ = "demo_agent_handoffs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("demo_calls.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(255), nullable=False)
    department: Mapped[str] = mapped_column(String(255), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    call: Mapped[DemoCallRecord] = relationship("DemoCallRecord", back_populates="handoffs")

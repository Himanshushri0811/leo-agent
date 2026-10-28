"""Additive migration: Add demo call center persistence tables.

Revision ID: 002_demo_call_tables
Revises: 001_initial_schema
Create Date: 2026-10-21 12:00:00.000000

Adds:
- demo_calls: stores simulated/live demo call records and lifecycle state
- demo_call_transcripts: stores ordered turns for each call session
- demo_agent_handoffs: stores escalation and human agent assignment records
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "002_demo_call_tables"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Demo Calls Table
    op.create_table(
        "demo_calls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("caller_id", sa.String(255), nullable=False),
        sa.Column("caller_name", sa.String(255), nullable=False),
        sa.Column("caller_phone_masked", sa.String(50), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="ringing"),
        sa.Column("is_muted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_simulated", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("auto_answer_delay_sec", sa.Integer(), nullable=False, server_default="4"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("escalated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("escalation_reason", sa.String(50), nullable=True),
        sa.Column("escalation_notes", sa.Text(), nullable=True),
        sa.Column("assigned_agent_id", sa.String(100), nullable=True),
        sa.Column("assigned_agent_name", sa.String(255), nullable=True),
        sa.Column("latest_error", sa.Text(), nullable=True),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_demo_calls_session_id", "demo_calls", ["session_id"])
    op.create_index("ix_demo_calls_status", "demo_calls", ["status"])
    op.create_index("ix_demo_calls_escalated", "demo_calls", ["escalated"])

    # 2. Demo Call Transcripts Table
    op.create_table(
        "demo_call_transcripts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "call_id",
            sa.String(36),
            sa.ForeignKey("demo_calls.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("sources", JSONB(), nullable=False, server_default="[]"),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_demo_call_transcripts_call_id", "demo_call_transcripts", ["call_id"])
    op.create_index("ix_demo_call_transcripts_call_turn", "demo_call_transcripts", ["call_id", "turn_index"])

    # 3. Demo Agent Handoffs Table
    op.create_table(
        "demo_agent_handoffs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "call_id",
            sa.String(36),
            sa.ForeignKey("demo_calls.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agent_id", sa.String(100), nullable=False),
        sa.Column("agent_name", sa.String(255), nullable=False),
        sa.Column("department", sa.String(255), nullable=False),
        sa.Column("reason", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_demo_agent_handoffs_call_id", "demo_agent_handoffs", ["call_id"])
    op.create_index("ix_demo_agent_handoffs_agent_id", "demo_agent_handoffs", ["agent_id"])


def downgrade() -> None:
    op.drop_table("demo_agent_handoffs")
    op.drop_table("demo_call_transcripts")
    op.drop_table("demo_calls")

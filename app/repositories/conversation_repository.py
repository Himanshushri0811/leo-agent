"""
Repository for Conversation and ConversationMessage entities.
Enforces strict session isolation by indexing and querying by session_id.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.conversation import Conversation, ConversationMessage, MessageRole


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, session_id: str) -> Conversation:
        """Find an existing conversation by session_id or create a new one."""
        stmt = (
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .options(selectinload(Conversation.messages))
        )
        result = await self._session.execute(stmt)
        conversation = result.scalar_one_or_none()

        if conversation is None:
            conversation = Conversation(session_id=session_id)
            self._session.add(conversation)
            await self._session.flush()
        return conversation

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: MessageRole | str,
        content: str,
        token_count: int | None = None,
        source_chunk_ids: list[str] | None = None,
    ) -> ConversationMessage:
        """Append a message turn to the specified conversation."""
        msg = ConversationMessage(
            conversation_id=conversation_id,
            role=str(role),
            content=content,
            token_count=token_count,
            source_chunk_ids=source_chunk_ids or [],
        )
        self._session.add(msg)
        await self._session.flush()
        return msg

    async def get_recent_messages(
        self, session_id: str, limit: int = 10
    ) -> Sequence[ConversationMessage]:
        """Fetch the most recent N messages for a given session."""
        stmt = (
            select(ConversationMessage)
            .join(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        # Reverse to return chronological order
        return list(reversed(result.scalars().all()))

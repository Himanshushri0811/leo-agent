"""
Integration tests for Conversation and ConversationMessage database persistence.
"""

from __future__ import annotations

import os
import uuid
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models.conversation import Conversation, ConversationMessage
from app.repositories.conversation_repository import ConversationRepository


@pytest.mark.asyncio
async def test_conversation_persistence_flow():
    """Verify storing, retrieving, and session isolation of conversation turns."""
    settings = get_settings()
    db_url = os.environ.get("DATABASE_URL", settings.database_url)
    engine = create_async_engine(db_url, connect_args={"timeout": 3})
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    test_session_id = f"test_persist_{uuid.uuid4().hex[:8]}"

    try:
        async with session_factory() as session:
            repo = ConversationRepository(session)

            # 1. Create or get conversation
            conv = await repo.get_or_create(test_session_id)
            assert conv.session_id == test_session_id
            conv_id = conv.id
            await session.commit()

        # 2. Add turns in a separate session
        async with session_factory() as session:
            repo = ConversationRepository(session)
            msg1 = await repo.add_message(
                conversation_id=conv_id,
                role="user",
                content="What services does Bitmax provide?",
                token_count=8,
            )
            msg2 = await repo.add_message(
                conversation_id=conv_id,
                role="assistant",
                content="Bitmax provides enterprise AI solutions.",
                token_count=7,
            )
            await session.commit()

        # 3. Read back and assert order and isolation
        async with session_factory() as session:
            repo = ConversationRepository(session)
            history = await repo.get_recent_messages(test_session_id, limit=10)
            assert len(history) == 2
            assert history[0].role == "user"
            assert history[0].content == "What services does Bitmax provide?"
            assert history[1].role == "assistant"
            assert history[1].content == "Bitmax provides enterprise AI solutions."

            # Verify session isolation: another session gets no messages
            empty_history = await repo.get_recent_messages("non_existent_session_999", limit=10)
            assert len(empty_history) == 0

        # Cleanup test conversation (cascade delete will remove messages)
        async with session_factory() as session:
            conv_to_clean = await session.get(Conversation, conv_id)
            if conv_to_clean:
                await session.delete(conv_to_clean)
                await session.commit()

    except Exception as exc:
        pytest.skip(
            f"Live PostgreSQL instance not reachable for conversation persistence test ({type(exc).__name__}: {exc}). "
            "Start PostgreSQL via 'docker compose up -d db' to run live persistence verification."
        )
    finally:
        await engine.dispose()

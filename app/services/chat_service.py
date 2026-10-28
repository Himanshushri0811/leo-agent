"""
Application business service for managing chat interactions and persistence.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.interfaces import Message
from app.ai.orchestrator import (
    ChatRequest as OrchChatRequest,
)
from app.ai.orchestrator import (
    ChatResponse as OrchChatResponse,
)
from app.ai.orchestrator import (
    run_chat,
)
from app.ai.providers import get_embeddings_provider, get_llm_provider
from app.core.config import Settings, get_settings
from app.db.models.conversation import MessageRole
from app.db.session import get_async_session
from app.observability.metrics import LLM_REQUESTS_TOTAL, LLM_TOKENS_TOTAL
from app.repositories.conversation_repository import ConversationRepository


class ChatService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._repo = ConversationRepository(session)
        self._llm = get_llm_provider(settings)
        self._embeddings = get_embeddings_provider(settings)

    async def process_chat(
        self,
        session_id: str,
        message: str,
        explicit_history: list[Message] | None = None,
    ) -> OrchChatResponse:
        conversation = await self._repo.get_or_create(session_id)

        if explicit_history:
            history = explicit_history
        else:
            db_msgs = await self._repo.get_recent_messages(session_id, limit=6)
            history = [Message(role=m.role, content=m.content) for m in db_msgs]

        orch_request = OrchChatRequest(
            session_id=session_id,
            user_message=message,
            conversation_history=history,
        )

        try:
            response = await run_chat(
                request=orch_request,
                llm=self._llm,
                embeddings=self._embeddings,
                settings=self._settings,
            )
            LLM_REQUESTS_TOTAL.labels(
                provider=response.provider,
                model=response.model,
                status="success",
            ).inc()
        except Exception:
            LLM_REQUESTS_TOTAL.labels(
                provider=self._llm.provider_name,
                model=self._settings.llm_model,
                status="error",
            ).inc()
            raise

        LLM_TOKENS_TOTAL.labels(provider=response.provider, type="prompt").inc(
            response.prompt_tokens
        )
        LLM_TOKENS_TOTAL.labels(provider=response.provider, type="completion").inc(
            response.completion_tokens
        )

        await self._repo.add_message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=message,
        )
        await self._repo.add_message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=response.response_text,
            token_count=response.completion_tokens,
            source_chunk_ids=[s.chunk_id for s in response.sources],
        )

        return response


def get_chat_service(
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> ChatService:
    return ChatService(session=session, settings=settings)

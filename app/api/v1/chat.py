"""
POST /api/v1/chat - customer-facing chat endpoint.

Accepts a customer question and optional conversation history.
Returns LEOs response with source attribution and request metadata.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.ai.interfaces import Message
from app.core.logging import get_logger
from app.core.security import check_request_size
from app.schemas.chat import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    SourceReference,
)
from app.services.chat_service import ChatService, get_chat_service

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = get_logger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(check_request_size)],
    summary="Customer chat",
    description=(
        "Send a customer question to LEO. Returns a grounded response with "
        "source references. Optionally include recent conversation history "
        "for multi-turn context."
    ),
)
async def chat(
    body: ChatRequest,
    request: Request,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    history = (
        [Message(role=m.role, content=m.content) for m in body.conversation_history]
        if body.conversation_history
        else None
    )

    result = await chat_service.process_chat(
        session_id=body.session_id,
        message=body.message,
        explicit_history=history,
    )

    return ChatResponse(
        session_id=result.session_id,
        response=result.response_text,
        sources=[
            SourceReference(
                chunk_id=s.chunk_id,
                source_name=s.source_name,
                source_url=s.source_url,
                similarity_score=s.similarity_score,
                content_excerpt=s.content_excerpt,
            )
            for s in result.sources
        ],
        metadata=ChatMetadata(
            model=result.model,
            provider=result.provider,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            prompt_version=result.prompt_version,
            sources_count=len(result.sources),
        ),
    )

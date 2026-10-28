"""Request/response schemas for the chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MessageContext(BaseModel):
    """A single previous turn for multi-turn context."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4096)


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Caller-generated session or conversation identifier.",
        examples=["sess_abc123"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer question.",
        examples=["What services does Bitmax Group offer?"],
    )
    conversation_history: list[MessageContext] = Field(
        default_factory=list,
        max_length=20,
        description="Recent conversation turns for multi-turn context (optional).",
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class SourceReference(BaseModel):
    chunk_id: str
    source_name: str
    source_url: str | None
    similarity_score: float
    content_excerpt: str


class ChatMetadata(BaseModel):
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    prompt_version: str
    sources_count: int


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[SourceReference]
    metadata: ChatMetadata

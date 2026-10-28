"""Request/response schemas for knowledge management endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import AnyHttpUrl, BaseModel, Field


class RegisterSourceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    source_url: AnyHttpUrl | None = None
    notes: str = Field(default="", max_length=1000)


class RegisterSourceResponse(BaseModel):
    source_id: uuid.UUID
    name: str
    status: str
    message: str


class IngestURLRequest(BaseModel):
    source_url: AnyHttpUrl
    name: str = Field(..., min_length=1, max_length=255)


class IngestResponse(BaseModel):
    source_id: uuid.UUID
    source_name: str
    chunks_created: int
    is_duplicate: bool
    status: str
    message: str


class ApproveSourceRequest(BaseModel):
    approved_by: str = Field(
        ..., min_length=1, max_length=255, examples=["reviewer@bitmaxgroup.com"]
    )


class KnowledgeSourceItem(BaseModel):
    id: uuid.UUID
    name: str
    source_type: str
    source_url: str | None
    status: str
    version: int
    created_at: datetime
    approved_at: datetime | None
    approved_by: str | None

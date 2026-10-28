"""Shared Pydantic schema types used across multiple API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    type: str
    message: str
    error_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthStatus(BaseModel):
    status: str  # "ok" | "degraded" | "unhealthy"
    version: str
    checks: dict[str, str] = Field(default_factory=dict)

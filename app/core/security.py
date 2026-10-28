"""
Security utilities: API key verification, request size limits, helpers.

Authentication strategy:
- Admin / write endpoints protected by a static API key passed in the
  X-API-Key header.
- Chat endpoint is customer-facing, decoupled from internal identities.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, AuthorizationError

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_admin_api_key(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency that enforces admin API key authentication."""
    if api_key is None:
        raise AuthenticationError("X-API-Key header is required.")
    if api_key != settings.admin_api_key.get_secret_value():
        raise AuthorizationError("Invalid API key.")


MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024  # 1 MB


async def check_request_size(request: Request) -> None:
    """Dependency that rejects requests whose declared Content-Length exceeds limit."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Request body too large.",
        )

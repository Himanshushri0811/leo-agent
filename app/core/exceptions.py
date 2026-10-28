"""
Domain exceptions and FastAPI exception handlers.

All handlers return safe, structured JSON responses.
Stack traces and internal details are never exposed to API consumers.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class LeoBaseError(Exception):
    """Base class for all LEO domain exceptions."""

    message: str = "An unexpected error occurred."
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str | None = None, **context: Any) -> None:
        self.message = message or self.__class__.message
        self.context = context
        super().__init__(self.message)


class NotFoundError(LeoBaseError):
    message = "Resource not found."
    status_code = status.HTTP_404_NOT_FOUND


class ValidationError(LeoBaseError):
    message = "Invalid input."
    status_code = 422


class AuthenticationError(LeoBaseError):
    message = "Authentication required."
    status_code = status.HTTP_401_UNAUTHORIZED


class AuthorizationError(LeoBaseError):
    message = "Insufficient permissions."
    status_code = status.HTTP_403_FORBIDDEN


class ProviderError(LeoBaseError):
    """Raised when an upstream AI or embedding provider call fails."""

    message = "AI provider error."
    status_code = status.HTTP_502_BAD_GATEWAY


class ProviderTimeoutError(ProviderError):
    message = "AI provider request timed out."
    status_code = status.HTTP_504_GATEWAY_TIMEOUT


class KnowledgeIngestionError(LeoBaseError):
    message = "Knowledge ingestion failed."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class PromptInjectionError(LeoBaseError):
    """Raised when retrieved content appears to contain injection attempts."""

    message = "Retrieved content was rejected due to policy."
    status_code = status.HTTP_400_BAD_REQUEST


def _error_body(request: Request, exc: LeoBaseError) -> dict[str, Any]:
    error_id = str(uuid.uuid4())
    logger.warning(
        "domain_error",
        error_id=error_id,
        error_type=type(exc).__name__,
        message=exc.message,
        path=str(request.url.path),
    )
    return {
        "error": {
            "type": type(exc).__name__,
            "message": exc.message,
            "error_id": error_id,
        }
    }


async def leo_exception_handler(request: Request, exc: LeoBaseError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(request, exc),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    error_id = str(uuid.uuid4())
    logger.exception(
        "unhandled_exception",
        error_id=error_id,
        path=str(request.url.path),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An internal error occurred. Please try again later.",
                "error_id": error_id,
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI application."""
    app.add_exception_handler(LeoBaseError, leo_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

"""
Health check endpoints.

GET /health/live  - liveness: always returns 200 if the process is running.
GET /health/ready - readiness: checks database connectivity.

These endpoints must NOT require authentication so load balancers and
container orchestrators can reach them freely.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.schemas.common import HealthStatus

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get(
    "/health/live",
    response_model=HealthStatus,
    summary="Liveness probe",
    description="Returns 200 if the application process is alive.",
)
async def liveness() -> HealthStatus:
    settings = get_settings()
    return HealthStatus(
        status="ok",
        version=settings.app_version,
    )


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Returns 200 when the application is ready to serve traffic. Checks DB.",
)
async def readiness() -> JSONResponse:
    settings = get_settings()
    checks: dict[str, str] = {}
    overall = "ok"

    # Database & pgvector check
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
            checks["database"] = "ok"

            ext_res = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            version_row = ext_res.scalar_one_or_none()
            if version_row:
                checks["pgvector"] = f"ok (v{version_row})"
            else:
                checks["pgvector"] = "not_installed"
                overall = "degraded"
    except Exception as exc:
        logger.warning("readiness_db_check_failed", error=str(exc))
        checks["database"] = "unavailable"
        checks["pgvector"] = "unknown"
        overall = "unhealthy"

    http_status = status.HTTP_200_OK if overall == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=http_status,
        content=HealthStatus(
            status=overall,
            version=settings.app_version,
            checks=checks,
        ).model_dump(),
    )

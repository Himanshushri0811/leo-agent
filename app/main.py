"""
LEO AI Backend - Main FastAPI Application Entrypoint.

Bootstraps logging, database lifecycle, security middleware, exception handlers,
Prometheus metrics, and versioned API routers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.db.session import dispose_db, init_db
from app.observability.middleware import (
    RequestCorrelationMiddleware,
    RequestLoggingMiddleware,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and graceful shutdown."""
    settings = get_settings()

    # 1. Logging setup
    configure_logging(
        log_level=settings.app_log_level,
        json_logs=settings.is_production,
    )
    logger.info(
        "app_startup",
        app_name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
        llm_provider=settings.llm_provider,
        embeddings_provider=settings.embeddings_provider,
    )

    # 2. Database connection pool initialization
    try:
        init_db()
        from app.services.demo_call_service import get_demo_service
        await get_demo_service().load_from_db()
    except Exception as exc:
        logger.error("database_init_failed", error=str(exc))
        if settings.is_production:
            raise

    yield

    # 3. Graceful shutdown
    logger.info("app_shutdown_started")
    await dispose_db()
    logger.info("app_shutdown_complete")


def create_application() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "LEO - Enterprise AI assistant backend for Bitmax Group. "
            "Provides grounded customer conversational Q&A, approved knowledge retrieval, "
            "and secure knowledge management."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # 1. Exception handlers
    register_exception_handlers(app)

    # 2. Observability & Correlation Middlewares (outermost first)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestCorrelationMiddleware)

    # 3. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 4. API Routers
    app.include_router(api_router)

    # 5. Dashboard static files & UI mount
    from pathlib import Path
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="dashboard")

        @app.get("/", include_in_schema=False)
        async def root() -> Response:
            return RedirectResponse(url="/dashboard/")

    # 6. Prometheus metrics endpoint (unauthenticated for scraper)
    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

    return app



app = create_application()

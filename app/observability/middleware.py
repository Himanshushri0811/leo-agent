"""
HTTP middleware for request correlation, structured access logging, and metrics.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger
from app.observability.metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL

logger = get_logger(__name__)


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request has a correlation/request ID.
    Propagates X-Request-ID (or generates a new UUID4) and binds it to structlog.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Bind request_id to contextvars for all subsequent log calls in this task
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs inbound requests and outbound responses with timing information.
    Records Prometheus metrics. Excludes request/response body for privacy and safety.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        path = request.url.path
        method = request.method

        # Skip high-frequency health / metrics logs in verbose stdout
        is_diagnostic = path.startswith(("/health", "/metrics"))

        if not is_diagnostic:
            logger.info("http_request_start", method=method, path=path)

        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start_time
            if not is_diagnostic:
                logger.info(
                    "http_request_end",
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_seconds=round(duration, 4),
                )

            # Record Prometheus metrics
            endpoint = self._get_route_pattern(request)
            HTTP_REQUESTS_TOTAL.labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint).observe(duration)

    @staticmethod
    def _get_route_pattern(request: Request) -> str:
        # Avoid high-cardinality route labels
        route = request.scope.get("route")
        if route and hasattr(route, "path"):
            return route.path
        return request.url.path

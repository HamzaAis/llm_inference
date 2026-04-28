from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.infrastructure.middleware.logger import setup_logger

logger = setup_logger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Measures the full request lifecycle from ASGI entry to response.

    This captures time spent in Pydantic validation, FastAPI dependency
    injection, and response serialization that individual handler timers
    cannot see.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        t0 = time.perf_counter()
        logger.info(
            "timing: request start method=%s path=%s",
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "timing: request end method=%s path=%s status=%d total_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Timing-Ms"] = f"{elapsed_ms:.2f}"
        return response

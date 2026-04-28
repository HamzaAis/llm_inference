from __future__ import annotations

import logging
import sys
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


_LOGGER_NAME = "llm_inferance.access"
_REQUEST_ID_HEADER = "x-request-id"


class RequestIDFilter(logging.Filter):
    def filter(self, record):
        from src.infrastructure.middleware.request_id import get_request_id
        record.request_id = get_request_id() or "no-request-id"
        return True


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        handler.addFilter(RequestIDFilter())

        formatter = logging.Formatter(
            '%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class AccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger_name: str = _LOGGER_NAME) -> None:
        super().__init__(app)
        self._logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER, uuid.uuid4().hex)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._logger.exception(
                "request failed method=%s path=%s duration_ms=%d request_id=%s",
                request.method,
                request.url.path,
                elapsed_ms,
                request_id,
            )
            raise
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers[_REQUEST_ID_HEADER] = request_id
        self._logger.info(
            "method=%s path=%s status=%d duration_ms=%d request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)

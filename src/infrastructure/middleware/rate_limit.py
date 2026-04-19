from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address


def build_limiter(per_minute: int) -> Limiter:
    return Limiter(
        key_func=get_remote_address,
        default_limits=[f"{per_minute}/minute"],
    )


def attach_rate_limiter(app: FastAPI, limiter: Limiter) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _handle_rate_limit_exceeded)
    app.add_middleware(SlowAPIMiddleware)


async def _handle_rate_limit_exceeded(request: Request, exc: RateLimitExceeded):
    return _rate_limit_exceeded_handler(request, exc)

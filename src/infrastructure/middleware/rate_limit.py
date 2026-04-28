from __future__ import annotations

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio
from src.infrastructure.config.settings import get_settings
from src.infrastructure.middleware.logger import setup_logger

logger = setup_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        settings = get_settings()
        self.requests = defaultdict(lambda: deque(maxlen=settings.rate_limit_per_minute))
        self.last_cleanup = datetime.now()
        self.rate_limit_enabled = settings.rate_limit_per_minute > 0
        self.rate_limit_requests_per_minute = settings.rate_limit_per_minute

    async def dispatch(self, request: Request, call_next):
        if not self.rate_limit_enabled:
            return await call_next(request)

        client_ip = request.client.host
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        if (now - self.last_cleanup).total_seconds() > 300:
            self._cleanup_old_ips(cutoff)
            self.last_cleanup = now

        ip_requests = self.requests[client_ip]

        while ip_requests and ip_requests[0] <= cutoff:
            ip_requests.popleft()

        if len(ip_requests) >= self.rate_limit_requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )

        ip_requests.append(now)

        response = await call_next(request)
        return response

    def _cleanup_old_ips(self, cutoff: datetime):
        ips_to_remove = [
            ip for ip, requests in self.requests.items()
            if not requests or (requests and requests[-1] <= cutoff)
        ]
        for ip in ips_to_remove:
            del self.requests[ip]
        if ips_to_remove:
            logger.debug(f"Cleaned up {len(ips_to_remove)} inactive IPs from rate limiter")

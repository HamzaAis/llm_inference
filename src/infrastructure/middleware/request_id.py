from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import contextvars

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default=None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))

        request_id_var.set(request_id)

        request.state.request_id = request_id

        response = await call_next(request)

        response.headers['X-Request-ID'] = request_id

        return response


def get_request_id() -> str:
    return request_id_var.get()

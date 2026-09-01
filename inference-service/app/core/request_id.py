"""Request correlation.

`docs/ARCHITECTURE.md` §15 specifies it: an ``X-Request-ID`` is accepted and
echoed, and generated when absent. This is the whole implementation — one
context variable, one middleware, and a formatter that reads it. No tracing
framework, no propagation machinery.
"""

import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-ID"
MAX_REQUEST_ID_LENGTH = 128
"""A caller-supplied id is echoed back, so it is bounded and stripped of
anything unprintable rather than trusted as-is."""

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def current_request_id() -> str:
    """The id of the request being handled, or empty outside one."""
    return _request_id.get()


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return "".join(char for char in value if char.isprintable())[:MAX_REQUEST_ID_LENGTH].strip()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Accept or generate a request id, expose it to logs, echo it back."""

    async def dispatch(self, request: Request, call_next):
        request_id = _clean(request.headers.get(REQUEST_ID_HEADER)) or str(uuid.uuid4())
        token = _request_id.set(request_id)
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

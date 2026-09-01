"""Wiring for the API layer.

FastAPI's own ``Depends`` is the only injection mechanism; no container and no
DI framework. The service is built once during startup and handed to every
request, so the model is never reloaded per request.
"""

from fastapi import Request

from app.application.inference_service import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    """Provide the service built at application startup."""
    return request.app.state.inference_service

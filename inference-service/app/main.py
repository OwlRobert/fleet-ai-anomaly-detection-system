"""Inference Service application factory."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import health, model, prediction
from app.application.inference_service import InferenceService
from app.core.config import Settings, get_settings
from app.core.errors import ArtifactLoadError
from app.core.logging import configure_logging
from app.core.request_id import RequestIdMiddleware
from app.infrastructure.artifact import load_artifact
from app.infrastructure.isolation_forest_model import IsolationForestAnomalyModel

logger = logging.getLogger(__name__)

DESCRIPTION = """
Anomaly scoring for EV fleet telemetry.

This service is **stateless and unit-agnostic**. It accepts canonical feature
values only — `soc` percent, `battery_voltage` V, `battery_current` A,
`battery_temperature` degC, `speed` km/h, `motor_rpm` rpm — and performs no unit
conversion. It knows nothing about vehicles, sites or clients.

The model is a scikit-learn **IsolationForest**, trained offline on synthetic
data by `ml/train.py` and loaded once at startup from a joblib artifact.
Training and serving are separate: this service never trains.

`anomaly_score` is anomaly-oriented — **higher means more anomalous**, with the
model's decision boundary at zero. It is a ranking score, not a probability.
"""


def build_inference_service(settings: Settings) -> InferenceService:
    """Load the artifact once and wrap it in the application service.

    A failure to load is not fatal. The process stays up, `/health` reports
    `model_loaded: false`, and the scoring endpoints answer `503`. Refusing to
    start would turn a bad artifact into an outage with no diagnostics.
    """
    try:
        artifact = load_artifact(
            settings.model_artifact_path,
            expected_name=settings.model_name,
            expected_version=settings.model_version,
        )
    except ArtifactLoadError as exc:
        # The path may appear here; that is fine in a log and never in a response.
        logger.error(
            "model artifact could not be loaded; this instance cannot serve predictions",
            extra={"detail": str(exc)},
        )
        return InferenceService(model=None)

    logger.info(
        "model artifact loaded",
        extra={
            "model_name": artifact.metadata.model_name,
            "model_version": artifact.metadata.model_version,
            "artifact_sha256": artifact.metadata.artifact_sha256[:12],
        },
    )
    return InferenceService(model=IsolationForestAnomalyModel(artifact))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once for the life of the process.

    Loading here rather than per request is the whole point: an IsolationForest
    is deserialized once and then reused by every prediction.
    """
    settings = get_settings()
    logger.info("inference service starting", extra={"version": settings.version})
    app.state.inference_service = build_inference_service(settings)
    try:
        yield
    finally:
        logger.info("inference service shutting down")


def create_app(*, lifespan_handler=lifespan) -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    app = FastAPI(
        title="Fleet Inference Service",
        version=settings.version,
        summary="Anomaly scoring over canonical EV telemetry features.",
        description=DESCRIPTION,
        lifespan=lifespan_handler,
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(prediction.router)
    return app


app = create_app()

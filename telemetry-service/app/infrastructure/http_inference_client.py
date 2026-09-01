"""The Inference Service, reached over HTTP.

This is the only module in the Telemetry Service that imports ``httpx``. Every
transport and contract failure is translated here into one application error
carrying an approved error code, so nothing above this layer sees a status
code, a URL or a driver exception.

A 200 is not trusted on its own: the body is validated before it becomes a
verdict. Persisting an unparseable response as ``COMPLETED`` would be a
fabricated result wearing a successful status.

One attempt, one bounded timeout, **no retries**. Retrying inside a synchronous
ingest request multiplies tail latency while the client waits, and the fail-open
policy already preserves the telemetry — a re-score can happen later from the
stored event.
"""

import logging
import math
from typing import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.application.errors import InferenceFailedError
from app.core.request_id import REQUEST_ID_HEADER, current_request_id
from app.domain.inference import InferenceErrorCode, InferenceOutcome
from app.domain.units import MetricName

logger = logging.getLogger(__name__)

PREDICT_PATH = "/predict"


class _PredictionBody(BaseModel):
    """The Inference Service's response contract, as we require it.

    Unknown fields are ignored so the upstream can add some without breaking
    us; the four we depend on are required and typed. ``allow_inf_nan=False``
    rejects a non-finite score, which would serialize to invalid JSON and is
    never a real verdict.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    is_anomaly: bool = Field(strict=True)
    anomaly_score: float = Field(strict=True, allow_inf_nan=False)
    model_name: str = Field(strict=True, min_length=1)
    model_version: str = Field(strict=True, min_length=1)


class HttpInferenceClient:
    """Calls ``POST {base_url}/predict`` with canonical features."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def predict(self, features: Mapping[MetricName, float]) -> InferenceOutcome:
        """Score one canonical feature vector.

        The payload carries the six canonical values and nothing else: no source
        units, no identifiers, no timestamps. The Inference Service is
        stateless and needs none of them.

        Raises:
            InferenceFailedError: On any transport, status or contract failure.
        """
        payload = {"features": {metric.value: float(value) for metric, value in features.items()}}

        # Carry the correlation id across the service boundary, so one id ties
        # an ingest request to the prediction it triggered.
        headers = {}
        request_id = current_request_id()
        if request_id:
            headers[REQUEST_ID_HEADER] = request_id

        try:
            response = await self._client.post(PREDICT_PATH, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise self._failure(InferenceErrorCode.TIMEOUT, "request timed out", exc) from exc
        except httpx.TransportError as exc:
            raise self._failure(InferenceErrorCode.UNREACHABLE, "service unreachable", exc) from exc
        except httpx.HTTPError as exc:  # any remaining httpx-level failure
            raise self._failure(InferenceErrorCode.UNREACHABLE, "request failed", exc) from exc

        if response.status_code >= 500:
            # Includes 503 MODEL_NOT_LOADED: the service is up but cannot score.
            raise self._failure(
                InferenceErrorCode.UNAVAILABLE, f"upstream returned {response.status_code}", None
            )
        if response.status_code != httpx.codes.OK:
            # A 4xx means the two contracts disagree about the request.
            raise self._failure(
                InferenceErrorCode.INVALID_RESPONSE,
                f"upstream rejected the request with {response.status_code}",
                None,
            )

        try:
            body = _PredictionBody.model_validate(response.json())
        except ValueError as exc:  # JSONDecodeError and ValidationError both
            detail = "response was not valid JSON"
            if isinstance(exc, ValidationError):
                detail = "response did not satisfy the prediction contract"
            raise self._failure(InferenceErrorCode.INVALID_RESPONSE, detail, exc) from exc

        if not math.isfinite(body.anomaly_score):  # pragma: no cover - allow_inf_nan covers it
            raise self._failure(
                InferenceErrorCode.INVALID_RESPONSE, "anomaly_score was not finite", None
            )

        return InferenceOutcome.completed(
            is_anomaly=body.is_anomaly,
            anomaly_score=body.anomaly_score,
            model_name=body.model_name,
            model_version=body.model_version,
        )

    @staticmethod
    def _failure(
        error_code: InferenceErrorCode, detail: str, cause: Exception | None
    ) -> InferenceFailedError:
        """Log what happened, then raise only the error code upward."""
        logger.warning(
            "inference call failed",
            extra={"error_code": error_code.value, "detail": detail},
            exc_info=cause is not None,
        )
        return InferenceFailedError(error_code, detail)


def create_inference_client(base_url: str, timeout_seconds: float) -> httpx.AsyncClient:
    """Build the process-wide HTTP client.

    One client, so the connection pool survives across requests. The timeout is
    explicit and finite: an unbounded wait would hold an ingest request open
    indefinitely, which fail-open exists to prevent.
    """
    return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_seconds))

"""Logging configuration.

The application attaches operational identifiers to log records through
``extra=``. The standard ``Formatter`` drops them, so without a formatter that
knows to look, ``event_id`` and friends are silently lost — which is what makes
a log line useless at exactly the moment it matters.

Both formatters here render those extras. ``json`` is the default, because
structured lines are what a log collector can actually query; ``text`` exists
for reading locally. Standard library only: no logging framework.
"""

import json
import logging
from typing import Any

from app.core.request_id import current_request_id

RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"asctime", "message", "taskName"}
"""Attributes every record has. Anything else came from ``extra=``."""


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """The fields the caller attached, and nothing else."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in RESERVED and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line, carrying whatever context the caller attached."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = current_request_id()
        if request_id:
            payload["request_id"] = request_id
        payload.update(_extras(record))
        if record.exc_info:
            # The traceback belongs in the log, never in a response body.
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable, with the same extras appended as key=value."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        context = _extras(record)
        request_id = current_request_id()
        if request_id:
            context = {"request_id": request_id, **context}
        if context:
            line += " " + " ".join(f"{key}={value}" for key, value in context.items())
        return line


def configure_logging(level: str, log_format: str) -> None:
    """Install one handler on the root logger.

    ``force=True`` replaces any handler a previous call installed, so building
    the application twice in one process (as the tests do) does not duplicate
    every line. Uvicorn configures its own loggers with ``propagate=False``, so
    its access log is untouched by this.
    """
    formatter = TextFormatter() if log_format.lower() == "text" else JsonFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)

"""Configuration that would make correct operation impossible is rejected.

These settings are read once at startup. A value that cannot work should stop
the process there, with a clear message, rather than surfacing later as a
confusing runtime failure.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_the_documented_defaults_are_valid() -> None:
    settings = Settings()

    assert settings.query_default_limit <= settings.query_max_limit
    assert settings.inference_timeout_seconds > 0
    assert settings.mongodb_timeout_seconds > 0


@pytest.mark.parametrize(
    ("label", "override"),
    [
        pytest.param("zero", {"INFERENCE_TIMEOUT_SECONDS": "0"}, id="inference-timeout-zero"),
        pytest.param("negative", {"INFERENCE_TIMEOUT_SECONDS": "-1"}, id="inference-timeout-negative"),
        pytest.param("zero", {"MONGODB_TIMEOUT_SECONDS": "0"}, id="mongo-timeout-zero"),
        pytest.param("negative", {"MONGODB_TIMEOUT_SECONDS": "-0.5"}, id="mongo-timeout-negative"),
    ],
)
def test_a_non_positive_timeout_is_rejected(label: str, override: dict) -> None:
    """An unbounded or nonsensical wait is exactly what the timeout exists to prevent."""
    with pytest.raises(ValidationError):
        Settings(**override)


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"TELEMETRY_QUERY_DEFAULT_LIMIT": "0"}, id="default-limit-zero"),
        pytest.param({"TELEMETRY_QUERY_MAX_LIMIT": "0"}, id="max-limit-zero"),
        pytest.param({"TELEMETRY_QUERY_DEFAULT_LIMIT": "-5"}, id="default-limit-negative"),
    ],
)
def test_a_non_positive_query_limit_is_rejected(override: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(**override)


def test_a_default_page_larger_than_the_maximum_is_rejected() -> None:
    """It could never be served, so it is a configuration error, not a runtime one."""
    with pytest.raises(ValidationError, match="must not exceed"):
        Settings(TELEMETRY_QUERY_DEFAULT_LIMIT="500", TELEMETRY_QUERY_MAX_LIMIT="100")


def test_equal_limits_are_allowed() -> None:
    settings = Settings(TELEMETRY_QUERY_DEFAULT_LIMIT="250", TELEMETRY_QUERY_MAX_LIMIT="250")

    assert settings.query_default_limit == settings.query_max_limit == 250


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"MONGODB_URI": ""}, id="empty-uri"),
        pytest.param({"MONGODB_DATABASE": ""}, id="empty-database"),
        pytest.param({"MONGODB_TELEMETRY_COLLECTION": ""}, id="empty-collection"),
    ],
)
def test_empty_store_configuration_is_rejected(override: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(**override)


@pytest.mark.parametrize(
    "url",
    ["", "not-a-url", "inference-service:8001", "ftp://inference-service:8001"],
)
def test_an_inference_url_without_a_usable_scheme_is_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(INFERENCE_SERVICE_URL=url)


@pytest.mark.parametrize(
    "url", ["http://inference-service:8001", "https://inference.internal", "http://127.0.0.1:8001"]
)
def test_usable_inference_urls_are_accepted(url: str) -> None:
    assert Settings(INFERENCE_SERVICE_URL=url).inference_service_url == url


def test_a_negative_clock_skew_bound_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(MAX_CLOCK_SKEW_FUTURE_SECONDS="-1")


def test_a_negative_event_age_bound_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(MAX_EVENT_AGE_DAYS="-1")


def test_zero_bounds_are_allowed_because_zero_is_meaningful() -> None:
    """Zero tolerance is a strict policy, not a broken one."""
    settings = Settings(MAX_CLOCK_SKEW_FUTURE_SECONDS="0", MAX_EVENT_AGE_DAYS="0")

    assert settings.max_clock_skew_future_seconds == 0
    assert settings.max_event_age_days == 0


@pytest.mark.parametrize("level", ["DEBUG", "info", "WARNING", "error", "CRITICAL"])
def test_valid_log_levels_are_accepted(level: str) -> None:
    assert Settings(LOG_LEVEL=level).log_level == level


@pytest.mark.parametrize("override", [{"LOG_LEVEL": "LOUD"}, {"LOG_FORMAT": "xml"}])
def test_invalid_logging_configuration_is_rejected(override: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(**override)


# --------------------------------------------------------------------------- #
# The connection string may carry credentials
# --------------------------------------------------------------------------- #


def test_the_connection_string_is_not_exposed_by_repr() -> None:
    settings = Settings(MONGODB_URI="mongodb://admin:hunter2@db.internal:27017")

    assert "hunter2" not in repr(settings)
    assert "hunter2" not in str(settings)
    assert "db.internal" not in repr(settings)


def test_the_connection_string_is_not_exposed_by_a_validation_error() -> None:
    """A neighbouring field failing must not print the credentials alongside it."""
    with pytest.raises(ValidationError) as raised:
        Settings(MONGODB_URI="mongodb://admin:hunter2@db.internal:27017", MONGODB_DATABASE="")

    assert "hunter2" not in str(raised.value)


def test_the_connection_string_is_still_readable_by_the_code_that_needs_it() -> None:
    settings = Settings(MONGODB_URI="mongodb://db.internal:27017")

    assert settings.mongodb_uri.get_secret_value() == "mongodb://db.internal:27017"

"""What makes two events the *same* logical event.

This single definition is what separates an idempotent retry from a conflicting
reuse of an `event_id`, so it is tested field by field.
"""

from datetime import timedelta

import pytest

from app.domain.units import MetricName
from tests.factories import CANONICAL_METRICS, FIXED_NOW, canonical_event


def test_an_identical_event_is_the_same_logical_event() -> None:
    assert canonical_event().is_same_logical_event(canonical_event())


def test_a_different_received_at_alone_does_not_make_a_retry_conflicting() -> None:
    """received_at is server-generated and differs on every retry by design."""
    first = canonical_event(received_at=FIXED_NOW)
    retry = canonical_event(received_at=FIXED_NOW + timedelta(minutes=17))

    assert first.is_same_logical_event(retry)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("vehicle_id", "veh-cz-0007", id="vehicle_id"),
        pytest.param("site_id", "site-prague-02", id="site_id"),
        pytest.param("schema_version", "1.1", id="schema_version"),
        pytest.param("event_id", "another-event", id="event_id"),
    ],
)
def test_a_different_logical_field_makes_the_event_different(field: str, value: str) -> None:
    assert not canonical_event().is_same_logical_event(canonical_event(**{field: value}))


def test_a_different_event_time_makes_the_event_different() -> None:
    other = canonical_event(event_time=FIXED_NOW - timedelta(hours=2))

    assert not canonical_event().is_same_logical_event(other)


def test_different_metric_values_make_the_event_different() -> None:
    other = canonical_event(metrics={**CANONICAL_METRICS, MetricName.SPEED: 0.0})

    assert not canonical_event().is_same_logical_event(other)


def test_different_source_units_make_the_event_different() -> None:
    """The same canonical number reported in a different unit is a different report."""
    other = canonical_event(source_units={**dict(canonical_event().source_units), MetricName.SPEED: "mph"})

    assert not canonical_event().is_same_logical_event(other)


def test_logical_identity_excludes_server_and_storage_concerns() -> None:
    identity = canonical_event().logical_identity

    assert FIXED_NOW not in identity  # received_at
    assert all(not isinstance(part, dict) for part in identity)


def test_logical_identity_is_order_independent_for_metrics() -> None:
    reordered = dict(reversed(list(CANONICAL_METRICS.items())))

    assert canonical_event().is_same_logical_event(canonical_event(metrics=reordered))


# --------------------------------------------------------------------------- #
# Storage resolution
# --------------------------------------------------------------------------- #


def test_sub_millisecond_precision_does_not_make_a_retry_conflicting() -> None:
    """The store keeps milliseconds, so equality is judged at that resolution.

    Without this, a first write of ...481789Z would come back as ...481000Z and
    every retry of that event would be misjudged as a conflict.
    """
    sent = canonical_event(event_time=FIXED_NOW.replace(microsecond=481789))
    read_back = canonical_event(event_time=FIXED_NOW.replace(microsecond=481000))

    assert sent.is_same_logical_event(read_back)


def test_a_millisecond_difference_still_makes_the_event_different() -> None:
    """Truncation must not blur genuinely different measurement times."""
    first = canonical_event(event_time=FIXED_NOW.replace(microsecond=481000))
    other = canonical_event(event_time=FIXED_NOW.replace(microsecond=482000))

    assert not first.is_same_logical_event(other)

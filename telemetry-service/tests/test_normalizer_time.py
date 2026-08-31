"""Time normalization, received_at, and the clock-skew bounds.

The clock is injected, so nothing here depends on when the suite runs.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.errors import ClockSkewFutureError, EventTooOldError, NormalizationError
from app.domain.normalizer import TelemetryNormalizer
from tests.factories import FIXED_NOW, source_event

MAX_FUTURE_SKEW = timedelta(seconds=300)
MAX_EVENT_AGE = timedelta(days=30)


def normalizer(now: datetime = FIXED_NOW) -> TelemetryNormalizer:
    return TelemetryNormalizer(
        clock=lambda: now, max_future_skew=MAX_FUTURE_SKEW, max_event_age=MAX_EVENT_AGE
    )


# --------------------------------------------------------------------------- #
# event_time -> UTC
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("source", "expected_utc"),
    [
        pytest.param(
            datetime(2026, 9, 1, 4, 30, tzinfo=timezone.utc),
            datetime(2026, 9, 1, 4, 30, tzinfo=timezone.utc),
            id="utc-unchanged",
        ),
        pytest.param(
            datetime(2026, 9, 1, 12, 30, tzinfo=timezone(timedelta(hours=8))),
            datetime(2026, 9, 1, 4, 30, tzinfo=timezone.utc),
            id="taipei-+08:00",
        ),
        pytest.param(
            datetime(2026, 9, 1, 8, 30, tzinfo=timezone(timedelta(hours=-7))),
            datetime(2026, 9, 1, 15, 30, tzinfo=timezone.utc),
            id="los-angeles--07:00",
        ),
        pytest.param(
            datetime(2026, 9, 1, 6, 30, tzinfo=timezone(timedelta(hours=2))),
            datetime(2026, 9, 1, 4, 30, tzinfo=timezone.utc),
            id="prague-summer-+02:00",
        ),
    ],
)
def test_event_time_is_converted_to_utc(source: datetime, expected_utc: datetime) -> None:
    canonical = normalizer().normalize(source_event(event_time=source))

    assert canonical.event_time == expected_utc


def test_conversion_preserves_the_instant() -> None:
    """Converting the offset must not move the moment in time."""
    source = datetime(2026, 9, 1, 12, 30, tzinfo=timezone(timedelta(hours=8)))

    canonical = normalizer().normalize(source_event(event_time=source))

    assert canonical.event_time.timestamp() == source.timestamp()


def test_local_wall_clock_is_never_reinterpreted_as_utc() -> None:
    """12:30+08:00 is 04:30Z, not 12:30Z."""
    source = datetime(2026, 9, 1, 12, 30, tzinfo=timezone(timedelta(hours=8)))

    canonical = normalizer().normalize(source_event(event_time=source))

    assert canonical.event_time.hour == 4
    assert canonical.event_time != source.replace(tzinfo=timezone.utc)


def test_normalized_event_time_is_timezone_aware_utc() -> None:
    canonical = normalizer().normalize(
        source_event(event_time=datetime(2026, 9, 1, 8, 30, tzinfo=timezone(timedelta(hours=-7))))
    )

    assert canonical.event_time.tzinfo is not None
    assert canonical.event_time.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------- #
# received_at
# --------------------------------------------------------------------------- #


def test_received_at_comes_from_the_server_clock() -> None:
    canonical = normalizer().normalize(
        source_event(event_time=FIXED_NOW - timedelta(minutes=30))
    )

    assert canonical.received_at == FIXED_NOW


def test_received_at_is_timezone_aware_utc() -> None:
    canonical = normalizer().normalize(source_event(event_time=FIXED_NOW))

    assert canonical.received_at.tzinfo is not None
    assert canonical.received_at.utcoffset() == timedelta(0)


def test_received_at_is_not_taken_from_the_event() -> None:
    """A delayed event keeps its own event_time and gets the server's arrival time."""
    event_time = FIXED_NOW - timedelta(hours=6)

    canonical = normalizer().normalize(source_event(event_time=event_time))

    assert canonical.event_time == event_time
    assert canonical.received_at == FIXED_NOW
    assert canonical.event_time != canonical.received_at


def test_event_time_and_received_at_stay_separate_fields() -> None:
    event_time = FIXED_NOW - timedelta(minutes=90)

    canonical = normalizer().normalize(source_event(event_time=event_time))

    assert canonical.ingest_delay == timedelta(minutes=90)


def test_the_clock_is_read_once_per_event() -> None:
    """A ticking clock must not produce two different received_at values."""
    ticks = iter([FIXED_NOW, FIXED_NOW + timedelta(seconds=5)])
    single_use = TelemetryNormalizer(
        clock=lambda: next(ticks), max_future_skew=MAX_FUTURE_SKEW, max_event_age=MAX_EVENT_AGE
    )

    canonical = single_use.normalize(source_event(event_time=FIXED_NOW - timedelta(minutes=1)))

    assert canonical.received_at == FIXED_NOW


# --------------------------------------------------------------------------- #
# Delayed, out-of-order, and skewed events
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "delay",
    [timedelta(seconds=1), timedelta(hours=6), timedelta(days=29, hours=23)],
    ids=["seconds", "hours", "just-inside-window"],
)
def test_delayed_events_normalize_successfully(delay: timedelta) -> None:
    """Buffered offline flushes are normal, not errors."""
    canonical = normalizer().normalize(source_event(event_time=FIXED_NOW - delay))

    assert canonical.ingest_delay == delay


def test_out_of_order_arrival_does_not_invalidate_an_event() -> None:
    """Normalization is stateless: processing order is not event-time order."""
    shared = normalizer()
    newer = shared.normalize(source_event(event_time=FIXED_NOW - timedelta(minutes=1)))
    older = shared.normalize(source_event(event_time=FIXED_NOW - timedelta(hours=3)))

    assert older.event_time < newer.event_time
    assert older.received_at == newer.received_at == FIXED_NOW


def test_event_exactly_at_the_future_skew_limit_is_accepted() -> None:
    """The bound is 'not more than', so the limit itself is inside it."""
    canonical = normalizer().normalize(source_event(event_time=FIXED_NOW + MAX_FUTURE_SKEW))

    assert canonical.event_time == FIXED_NOW + MAX_FUTURE_SKEW


def test_event_beyond_the_future_skew_limit_is_rejected() -> None:
    with pytest.raises(ClockSkewFutureError) as raised:
        normalizer().normalize(
            source_event(event_time=FIXED_NOW + MAX_FUTURE_SKEW + timedelta(seconds=1))
        )

    assert raised.value.limit == MAX_FUTURE_SKEW
    assert raised.value.received_at == FIXED_NOW


def test_event_exactly_at_the_max_age_is_accepted() -> None:
    canonical = normalizer().normalize(source_event(event_time=FIXED_NOW - MAX_EVENT_AGE))

    assert canonical.event_time == FIXED_NOW - MAX_EVENT_AGE


def test_event_older_than_the_max_age_is_rejected() -> None:
    with pytest.raises(EventTooOldError) as raised:
        normalizer().normalize(
            source_event(event_time=FIXED_NOW - MAX_EVENT_AGE - timedelta(seconds=1))
        )

    assert raised.value.limit == MAX_EVENT_AGE


def test_skew_rejections_are_normalization_errors() -> None:
    for event_time in (FIXED_NOW + timedelta(days=1), FIXED_NOW - timedelta(days=365)):
        with pytest.raises(NormalizationError):
            normalizer().normalize(source_event(event_time=event_time))


def test_skew_is_measured_against_the_utc_instant_not_the_local_reading() -> None:
    """A +08:00 event 30 minutes old must not look 8.5 hours into the future."""
    source = (FIXED_NOW - timedelta(minutes=30)).astimezone(timezone(timedelta(hours=8)))

    canonical = normalizer().normalize(source_event(event_time=source))

    assert canonical.ingest_delay == timedelta(minutes=30)


def test_a_rejected_timestamp_is_never_clamped_or_rewritten() -> None:
    """The event is refused; its timestamp is reported back unchanged."""
    too_far = FIXED_NOW + timedelta(days=2)

    with pytest.raises(ClockSkewFutureError) as raised:
        normalizer().normalize(source_event(event_time=too_far))

    assert raised.value.event_time == too_far

"""Range, ordering and limit semantics of the two history endpoints."""

from datetime import timedelta
from fastapi.testclient import TestClient

from app.domain.inference import InferenceErrorCode, InferenceOutcome
from tests.factories import FIXED_NOW, scored, stored_event
from tests.fakes import InMemoryTelemetryRepository

VEHICLE = "veh-tw-0142"
HISTORY_URL = f"/api/v1/vehicles/{VEHICLE}/telemetry"
ANOMALIES_URL = f"/api/v1/vehicles/{VEHICLE}/anomalies"


def at(offset: timedelta):
    return FIXED_NOW + offset


def window(before: timedelta = timedelta(days=7), after: timedelta = timedelta(days=1)):
    return {"start": (FIXED_NOW - before).isoformat(), "end": (FIXED_NOW + after).isoformat()}


def ids(response) -> list[str]:
    return [item["event_id"] for item in response.json()["items"]]


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_results_are_newest_event_time_first(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.extend(
        [
            stored_event(event_id="oldest", event_time=at(timedelta(hours=-9))),
            stored_event(event_id="newest", event_time=at(timedelta(hours=-1))),
            stored_event(event_id="middle", event_time=at(timedelta(hours=-5))),
        ]
    )

    assert ids(client.get(HISTORY_URL, params=window())) == ["newest", "middle", "oldest"]


def test_insertion_order_does_not_affect_result_order(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    """A late-arriving event lands in its temporal position, not at the end."""
    repository.events.append(stored_event(event_id="arrived-first", event_time=at(timedelta(hours=-1))))
    repository.events.append(stored_event(event_id="arrived-late", event_time=at(timedelta(hours=-6))))

    assert ids(client.get(HISTORY_URL, params=window())) == ["arrived-first", "arrived-late"]


def test_ordering_ignores_received_at(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.extend(
        [
            stored_event(
                event_id="older-event-newer-arrival",
                event_time=at(timedelta(hours=-8)),
                received_at=FIXED_NOW,
            ),
            stored_event(
                event_id="newer-event-older-arrival",
                event_time=at(timedelta(hours=-2)),
                received_at=FIXED_NOW - timedelta(hours=1),
            ),
        ]
    )

    assert ids(client.get(HISTORY_URL, params=window()))[0] == "newer-event-older-arrival"


# --------------------------------------------------------------------------- #
# Range boundaries: [start, end)
# --------------------------------------------------------------------------- #


def test_start_is_inclusive_and_end_is_exclusive(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    start, end = at(timedelta(hours=-4)), at(timedelta(hours=-1))
    repository.events.extend(
        [
            stored_event(event_id="before", event_time=start - timedelta(seconds=1)),
            stored_event(event_id="on-start", event_time=start),
            stored_event(event_id="inside", event_time=start + timedelta(minutes=30)),
            stored_event(event_id="on-end", event_time=end),
            stored_event(event_id="after", event_time=end + timedelta(seconds=1)),
        ]
    )

    response = client.get(HISTORY_URL, params={"start": start.isoformat(), "end": end.isoformat()})

    assert set(ids(response)) == {"on-start", "inside"}


def test_an_empty_range_returns_an_empty_page(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.append(stored_event(event_id="outside", event_time=at(timedelta(days=-3))))

    page = client.get(
        HISTORY_URL,
        params={"start": at(timedelta(hours=-1)).isoformat(), "end": FIXED_NOW.isoformat()},
    ).json()

    assert page == {
        "vehicle_id": VEHICLE,
        "start": page["start"],
        "end": page["end"],
        "count": 0,
        "items": [],
    }


# --------------------------------------------------------------------------- #
# Vehicle filter and limit
# --------------------------------------------------------------------------- #


def test_only_the_requested_vehicle_is_returned(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.extend(
        [
            stored_event(event_id="ours", vehicle_id=VEHICLE),
            stored_event(event_id="theirs", vehicle_id="veh-cz-0007"),
        ]
    )

    assert ids(client.get(HISTORY_URL, params=window())) == ["ours"]


def test_the_limit_bounds_the_page(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.extend(
        stored_event(event_id=f"evt-{index}", event_time=at(timedelta(minutes=-index)))
        for index in range(1, 11)
    )

    page = client.get(HISTORY_URL, params=window() | {"limit": 3}).json()

    assert page["count"] == 3
    assert [item["event_id"] for item in page["items"]] == ["evt-1", "evt-2", "evt-3"]


def test_the_limit_keeps_the_newest_events(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    """Truncation happens after ordering, so a page is the newest N."""
    repository.events.extend(
        stored_event(event_id=f"evt-{index}", event_time=at(timedelta(hours=-index)))
        for index in range(1, 6)
    )

    assert ids(client.get(HISTORY_URL, params=window() | {"limit": 2})) == ["evt-1", "evt-2"]


# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #


def test_only_confirmed_anomalies_are_returned(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    """Storage fixtures for future scored documents. No model produced these."""
    repository.events.extend(
        [
            stored_event(event_id="anomalous", inference=scored(is_anomaly=True)),
            stored_event(event_id="scored-normal", inference=scored(is_anomaly=False)),
            stored_event(event_id="unscored"),
        ]
    )

    assert ids(client.get(ANOMALIES_URL, params=window())) == ["anomalous"]


def test_unscored_events_are_never_treated_as_non_anomalous(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    """A missing verdict is not a negative verdict, in either direction."""
    repository.events.append(stored_event(event_id="unscored"))

    assert client.get(ANOMALIES_URL, params=window()).json()["count"] == 0
    assert client.get(HISTORY_URL, params=window()).json()["count"] == 1


def test_anomalies_respect_ordering_range_and_limit(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.extend(
        [
            stored_event(event_id="a-old", event_time=at(timedelta(hours=-9)), inference=scored(True)),
            stored_event(event_id="a-new", event_time=at(timedelta(hours=-1)), inference=scored(True)),
            stored_event(event_id="a-out", event_time=at(timedelta(days=-30)), inference=scored(True)),
        ]
    )

    assert ids(client.get(ANOMALIES_URL, params=window())) == ["a-new", "a-old"]
    assert ids(client.get(ANOMALIES_URL, params=window() | {"limit": 1})) == ["a-new"]


def test_anomalies_filter_by_vehicle(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.extend(
        [
            stored_event(event_id="ours", vehicle_id=VEHICLE, inference=scored(True)),
            stored_event(event_id="theirs", vehicle_id="veh-cz-0007", inference=scored(True)),
        ]
    )

    assert ids(client.get(ANOMALIES_URL, params=window())) == ["ours"]


def test_failed_and_pending_events_are_excluded_from_anomalies(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    """Only a completed run's `true` counts. Absence of a verdict is not a verdict."""
    repository.events.extend(
        [
            stored_event(event_id="anomalous", inference=scored(is_anomaly=True)),
            stored_event(event_id="scored-normal", inference=scored(is_anomaly=False)),
            stored_event(event_id="failed", inference=InferenceOutcome.failed(InferenceErrorCode.TIMEOUT)),
            stored_event(event_id="legacy-pending"),
        ]
    )

    assert ids(client.get(ANOMALIES_URL, params=window())) == ["anomalous"]


def test_history_returns_every_state(
    client: TestClient, repository: InMemoryTelemetryRepository
) -> None:
    repository.events.extend(
        [
            stored_event(event_id="anomalous", inference=scored(is_anomaly=True)),
            stored_event(event_id="scored-normal", inference=scored(is_anomaly=False)),
            stored_event(event_id="failed", inference=InferenceOutcome.failed(InferenceErrorCode.UNAVAILABLE)),
            stored_event(event_id="legacy-pending"),
        ]
    )

    statuses = {
        item["event_id"]: item["inference"]["status"]
        for item in client.get(HISTORY_URL, params=window()).json()["items"]
    }

    assert statuses == {
        "anomalous": "COMPLETED",
        "scored-normal": "COMPLETED",
        "failed": "FAILED",
        "legacy-pending": "PENDING",
    }

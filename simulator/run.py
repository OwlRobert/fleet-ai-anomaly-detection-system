"""Telemetry simulator — a demo client for the Telemetry Service.

Sends synthetic telemetry for a small fleet of EVs, mostly ordinary readings
with an occasional deliberately extreme one, and prints what the service
actually answered.

    python simulator/run.py                  # continuous, Ctrl+C to stop
    python simulator/run.py --count 20       # send 20 events and exit
    python simulator/run.py --seed 42        # reproducible run

Standard library only: no install step, no virtualenv, nothing to build. It
talks to the Telemetry Service over HTTP and **never** to the Inference Service
— routing telemetry to the model is the Telemetry Service's job, and the
simulator has no business knowing the model exists.

It is a client, not a component: it holds no retry logic, no queue and no
buffering. When the service is unreachable it says so and carries on.
"""

import argparse
import json
import os
import random
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    # Running as `python simulator/run.py` puts this directory on the path, not
    # the repository root. Add the root so the package import below resolves,
    # and so both this and `python -m simulator.run` work.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.events import generate_event, summarize_response, vehicle_ids

DEFAULT_TARGET_URL = os.environ.get(
    "SIMULATOR_TARGET_URL", "http://localhost:8000/api/v1/telemetry"
)
DEFAULT_VEHICLE_COUNT = int(os.environ.get("SIMULATOR_VEHICLE_COUNT", "3"))
DEFAULT_INTERVAL_SECONDS = float(os.environ.get("SIMULATOR_INTERVAL_SECONDS", "1.0"))
DEFAULT_SITE_TIMEZONE = os.environ.get("SIMULATOR_SITE_TIMEZONE", "Asia/Taipei")
DEFAULT_SITE_ID = "site-taipei-01"
DEFAULT_ANOMALY_RATE = 0.15
REQUEST_TIMEOUT_SECONDS = 15.0
"""Bounded, so an unresponsive service cannot wedge the simulator.

Deliberately longer than the Telemetry Service's own worst case — it waits up
to `MONGODB_TIMEOUT_SECONDS` for the store before answering `503`. A shorter
client timeout would hide that answer behind a client-side timeout, which is
the opposite of what the failure demo is meant to show."""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_TARGET_URL, help="Telemetry ingest endpoint.")
    parser.add_argument("--vehicles", type=int, default=DEFAULT_VEHICLE_COUNT, help="Fleet size.")
    parser.add_argument("--site-id", default=DEFAULT_SITE_ID, help="Site the fleet reports from.")
    parser.add_argument(
        "--timezone", default=DEFAULT_SITE_TIMEZONE, help="IANA zone for event_time offsets."
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_SECONDS, help="Seconds between events."
    )
    parser.add_argument(
        "--count", type=int, default=0, help="Number of events to send; 0 runs until Ctrl+C."
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=DEFAULT_ANOMALY_RATE,
        help="Share of events generated as anomaly candidates, 0.0-1.0.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for a reproducible run.")
    return parser.parse_args(argv)


def post_event(url: str, payload: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None, str]:
    """Send one event.

    Returns:
        ``(status, body, note)``. On a transport failure ``status`` is ``None``
        and ``note`` explains it in one line — no traceback, because a service
        that is not running is an expected demo condition, not a crash.
    """
    request = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, json.load(response), ""
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.load(exc), ""
        except ValueError:
            return exc.code, None, "response was not JSON"
    except urllib.error.URLError as exc:
        return None, None, f"telemetry service unreachable ({exc.reason})"
    except TimeoutError:
        return None, None, "telemetry service did not answer in time"


def run(arguments: argparse.Namespace) -> int:
    """Send events until the count is reached or the user interrupts."""
    if not 0.0 <= arguments.anomaly_rate <= 1.0:
        print("--anomaly-rate must be between 0.0 and 1.0", file=sys.stderr)
        return 2

    try:
        site_timezone = ZoneInfo(arguments.timezone)
    except Exception:  # noqa: BLE001 - a bad zone name is user error, not a crash
        print(f"unknown timezone {arguments.timezone!r}", file=sys.stderr)
        return 2

    rng = random.Random(arguments.seed)
    fleet = vehicle_ids(arguments.vehicles)

    print(f"telemetry simulator -> {arguments.url}")
    print(f"  fleet {', '.join(fleet)} at {arguments.site_id} ({arguments.timezone})")
    print(f"  every {arguments.interval}s, anomaly candidates ~{arguments.anomaly_rate:.0%}")
    print(f"  {'continuous, Ctrl+C to stop' if not arguments.count else f'{arguments.count} events'}\n")

    sent = accepted = rejected = undelivered = 0
    try:
        while not arguments.count or sent < arguments.count:
            event = generate_event(
                vehicle_id=fleet[sent % len(fleet)],
                site_id=arguments.site_id,
                rng=rng,
                site_timezone=site_timezone,
                inject_anomaly=rng.random() < arguments.anomaly_rate,
            )
            status, body, note = post_event(arguments.url, event.payload)
            sent += 1

            intent = "injected" if event.injected else "normal  "
            if status is None:
                undelivered += 1
                print(f"{event.vehicle_id} | {intent} | {note}")
            else:
                # 2xx means the telemetry was stored; anything else means it was
                # not, whatever the inference outcome was.
                if 200 <= status < 300:
                    accepted += 1
                else:
                    rejected += 1
                print(f"{event.vehicle_id} | {intent} | {summarize_response(status, body)}")

            if not arguments.count or sent < arguments.count:
                time.sleep(arguments.interval)
    except KeyboardInterrupt:
        print()

    print(
        f"\nsent {sent} event(s): {accepted} accepted, "
        f"{rejected} rejected by the service, {undelivered} undelivered"
    )
    return 0


def main() -> int:
    # Restore default Ctrl+C handling; KeyboardInterrupt is caught in run().
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return run(parse_arguments())


if __name__ == "__main__":
    raise SystemExit(main())

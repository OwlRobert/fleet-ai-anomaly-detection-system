# Architecture Decision Records

Concise ADRs for **fleet-ai-anomaly-detection-system**. Each record states the decision, why it was
taken, what it costs, what was rejected, and the signal that should trigger revisiting it.

All records below are **Accepted** as of 2026-08-31. Where a decision is already implemented, the
record says so; the rest describe behavior later phases build.

| ADR | Title |
| --- | --- |
| [0001](#adr-0001-mongodb-as-the-mvp-telemetry-event-store) | MongoDB as the MVP telemetry event store |
| [0002](#adr-0002-synchronous-http-inference-for-the-mvp) | Synchronous HTTP inference for the MVP |
| [0003](#adr-0003-canonical-internal-telemetry-units) | Canonical internal telemetry units |
| [0004](#adr-0004-timezone-aware-input-utc-persistence-iana-site-timezones) | Timezone-aware input, UTC persistence, IANA site timezones |
| [0005](#adr-0005-event_id-as-the-idempotency-key) | `event_id` as the idempotency key |
| [0006](#adr-0006-fail-open-persistence-when-inference-fails) | Fail-open persistence when inference fails; fail-closed ingestion when persistence fails |
| [0007](#adr-0007-transport-independent-ingestion-use-case) | Transport-independent ingestion use case |
| [0008](#adr-0008-deliberately-limited-ports) | Deliberately limited ports |
| [0009](#adr-0009-pending-as-the-unscored-inference-state) | `PENDING` as the unscored inference state |
| [0010](#adr-0010-train-the-model-during-the-image-build) | Train the model during the image build |

---

## ADR-0001: MongoDB as the MVP telemetry event store

**Status:** Accepted · 2026-08-31

### Context

The system stores telemetry events from a multinational EV fleet. Events are immutable, arrive
continuously, and are read back by vehicle and time range. Device and firmware generations across
regions do not evolve their metric sets in lockstep. The project also has an explicit learning
objective: exercise a NoSQL persistence model.

### Decision

Use **MongoDB** as the telemetry event store for the MVP, with a single `telemetry_events`
collection behind the `TelemetryRepository` port.

The reasoning is:

* Telemetry is **append-heavy, immutable event documents** — no updates, no joins in the MVP read
  paths. A document store's write path and access model match that shape directly.
* **Metric sets evolve per device/firmware generation.** A per-document schema absorbs that
  without coordinated migrations across a heterogeneous fleet.
* The **access patterns are narrow and compound** — vehicle + time range, anomaly + time range —
  and are served by a compound index and a partial index on the anomaly flag.
* The project **intentionally exercises NoSQL**, which is a legitimate reason for a learning
  project and is stated as such rather than dressed up as a technical necessity.

**Explicitly rejected reasoning:** *"the payload is JSON, so use a document database."* Wire format
is not a storage argument. PostgreSQL stores and indexes JSON natively in `JSONB` and would handle
this payload without difficulty.

### Alternative: PostgreSQL with JSONB

A fully viable alternative for this system. It would be **preferable** when:

| Condition | Why PostgreSQL wins |
| --- | --- |
| The metric schema stabilizes | Column types, `NOT NULL`, `CHECK` constraints enforced by the database rather than by application code |
| Relational requirements appear | Vehicles, sites, models, alerts, maintenance records as first-class entities with foreign keys and referential integrity |
| Queries need joins | Native joins across those entities instead of denormalization or client-side assembly |
| Workflows need transactions | Mature multi-row/multi-table ACID semantics |
| Analytics get complex | Window functions, CTEs, percentile aggregates — markedly more expressive than an aggregation pipeline for this class of query |
| Operational simplicity matters | One engine for both relational and document data |

Neither database is inherently superior. MongoDB fits this MVP's shape and stated goals;
PostgreSQL would fit a relationally-richer version of the same system.

### Consequences

* No schema enforcement from the database; validation is the application's responsibility.
* Relational features arriving later (alerts, registries, maintenance history) will be more awkward
  than they would be in PostgreSQL.
* Storage is confined behind `TelemetryRepository`, so the code cost of switching is one adapter —
  though a real migration also involves data movement and index redesign, not just new code.

### Revisit when

Relational entities and joins become central, the metric schema stabilizes, or analytics outgrow
the aggregation pipeline.

---

## ADR-0002: Synchronous HTTP inference for the MVP

**Status:** Accepted · 2026-08-31

### Context

Each telemetry event must be scored for anomaly. The model runs in a separate FastAPI service. The
options are a synchronous HTTP call inside the ingest request, or asynchronous scoring via a
message broker and worker.

### Decision

The Telemetry Service calls the Inference Service over **synchronous HTTP** inside the ingest
request, with a bounded timeout (`INFERENCE_TIMEOUT_SECONDS`, default 2 s) and **no in-request
retries**.

Rationale: the ingest response can report the anomaly result immediately, which makes the system
easy to demonstrate and reason about; there is no broker, worker, or delivery-semantics machinery
to build or explain; and the failure mode is explicit and bounded rather than hidden in a queue.
The MVP's throughput is a simulator, so tail latency is not a real constraint yet.

No in-request retries: retrying inside a synchronous request multiplies tail latency while the
client is blocked, and the client's own retry — safe by idempotency ([ADR-0005](#adr-0005-event_id-as-the-idempotency-key)) — already covers the transient case.

### Consequences

* Ingest latency includes inference latency, and is bounded by the timeout.
* Inference availability affects ingest *quality*, never ingest *durability* — inference failure is
  fail-open for telemetry persistence
  ([ADR-0006](#adr-0006-fail-open-persistence-when-inference-fails)).
* Throughput is coupled to the slower of the two services.
* Inference is attempted before the single persistence write, so each event is written once.

### Alternatives considered

* **Queue-based async scoring** — better throughput isolation and back-pressure, at the cost of a
  broker, a worker, delivery-semantics design, and a two-phase persistence shape. Deferred.
* **In-process model** — removes the network hop, but also removes the inference boundary, the
  versioning surface, and the failure mode this project exists to demonstrate.

### Migration path

`InferencePort` hides the mechanism. Moving to a queue changes the adapter and the persistence
shape — write first with `status = PENDING`, update on completion — but **does not change the
ingest transport contract**. Clients keep posting the same payload and simply stop receiving a
score in the response.

### Revisit when

Ingest throughput is constrained by inference latency, or inference needs to scale independently.

---

## ADR-0003: Canonical internal telemetry units

**Status:** Accepted · 2026-08-31

### Context

Vehicles across regions report measurements in different units, and a single device commonly mixes
conventions — `mph` for speed alongside `degC` for battery temperature, or `mV` from a
high-resolution sensor. Storing values in whatever unit arrived would make every stored number
uninterpretable without its unit, and would make the model's input depend on the client.

### Decision

* External telemetry declares units **per metric**, as `{ "value": …, "unit": … }`. There is no
  request-level `"metric"` / `"imperial"` flag — units are a property of each measurement.
* Units are **always explicit**. An omitted unit is a validation error, never an assumed default.
* `TelemetryNormalizer` converts to **canonical units once, at the ingestion boundary**, before
  persistence and before inference: `soc` in `percent`, voltage in `V`, current in `A`, temperature
  in `degC`, speed in `km/h`, motor speed in `rpm`.
* Stored `metrics` are **canonical scalars**. Source units are kept separately as provenance
  (`source_units`) and are never used for querying or scoring.
* Unsupported units are **rejected** (`UNSUPPORTED_UNIT`), never passed through unconverted.
* The Inference Service receives canonical values only and performs no conversion.

### Consequences

* Every stored value is comparable across the fleet without carrying a unit.
* The model never depends on client display preferences or source measurement units, so its
  behavior does not change with the caller's locale.
* Conversion bugs are confined to one component with one table.
* Clients must send explicit units, which is slightly more verbose — accepted, because a
  silently-assumed unit is precisely the failure this contract prevents.
* Adding a new accepted unit is a table entry; adding a new *canonical* unit is a breaking change
  affecting stored data and the model.

### Alternatives considered

* **Request-level unit system flag** — wrong for the domain: real firmware mixes conventions within
  one device.
* **Store as received, convert on read** — pushes conversion into every reader and every query, and
  makes range predicates on stored values meaningless.

---

## ADR-0004: Timezone-aware input, UTC persistence, IANA site timezones

**Status:** Accepted · 2026-08-31

### Context

Sites span multiple countries and timezones, several of which observe daylight saving time. Devices
have imperfect clocks and intermittent connectivity, so measurement time and arrival time differ.

### Decision

* Incoming `event_time` must be **timezone-aware ISO-8601**; naive timestamps are **rejected**
  (`NAIVE_TIMESTAMP`).
* `event_time` is normalized to **UTC** for storage and querying.
* The Telemetry Service stamps **`received_at`** in UTC on arrival; it is never accepted from the
  client.
* Queries order primarily by **`event_time`**, not by insertion order.
* Any site timezone stored as metadata later must use **IANA names** (`Asia/Taipei`,
  `Europe/Prague`, `America/Los_Angeles`), **never fixed UTC offsets**.
* Clock skew is validated against `received_at` with bounded tolerances and distinct rejection
  codes (`CLOCK_SKEW_FUTURE`, `EVENT_TOO_OLD`); the MVP rejects rather than corrects.

### Rationale

* A naive timestamp is ambiguous across the fleet's regions. Guessing "probably UTC" or "probably
  the site's local time" produces data that is silently wrong and unfixable after the fact.
* Two timestamps answer different questions: `event_time` is physically meaningful (when the
  measurement happened), `received_at` is operationally meaningful (delivery behavior, measured on
  one trusted clock). Keeping only one loses either delivery visibility or the true time-series.
* `+01:00` is not a timezone. Prague is `+01:00` in winter and `+02:00` in summer; a stored offset
  breaks twice a year at the DST transition, and reinterpreting historical data afterwards is
  expensive. IANA names carry the full rule set, including historical and future changes.
* Rewriting a device's timestamp to server time would destroy the evidence that its clock is wrong
  and fabricate a measurement time.

### Consequences

* Clients must send offsets; a common source of silent corruption is eliminated at the boundary.
* Local wall-clock display is a client concern, reconstructed from an IANA timezone when needed.
* The original offset is not preserved in the MVP.
* BSON dates give millisecond precision in UTC; sub-millisecond device precision is not retained —
  acceptable at second-scale sampling.
* Skew rejection means a badly-desynchronized device loses events rather than polluting the store.
  A production system would quarantine and alert instead; the rejection codes are the seam for that.

---

## ADR-0005: `event_id` as the idempotency key

**Status:** Accepted (behavior specified; **mechanism not implemented** in the MVP) · 2026-08-31

### Context

Telemetry delivery is at-least-once. A client that times out cannot know whether the server stored
the event, so it retries. MQTT QoS 1, added later, has the same property by protocol. Without
idempotency, every timeout inflates the fleet's history with duplicates that corrupt counts and
time-series.

### Decision

`event_id` is **globally unique per emitted telemetry event**, generated at emission and **stable
across retries**. It is the idempotency key for ingestion.

Intended behavior:

| Situation | Result |
| --- | --- |
| First arrival | Stored; `201 Created`; `duplicate: false` |
| Retry with the same `event_id` | No second record, no second inference call; `200 OK`; `duplicate: true`; stored record returned |
| Same `event_id`, different payload | Rejected by the uniqueness constraint. The originally stored event is **never overwritten or updated**; the conflict is logged with the differing fields and the stored original is returned. `409` instead is a defensible alternative, deferred until there is evidence it occurs |

### Domain identity is independent of storage identity

`event_id` is a **domain** identifier, not the storage engine's primary key. The two are kept
separate deliberately:

* `event_id` is the globally unique domain-level telemetry event identifier and the idempotency key
  exposed by the API contract.
* The persistence engine keeps whatever internal identity it likes. MongoDB assigns its ordinary
  `_id` (`ObjectId`); the application never reads it, never returns it, and it never crosses the
  `TelemetryRepository` boundary.

Intended mechanism: a **unique index on `event_id`**. Uniqueness is enforced by that constraint,
and a duplicate insert surfaces as a uniqueness violation that the use case translates into the
`duplicate` response — checked by the database rather than by a read-then-write race in application
code.

A lookup by `event_id` before calling inference is permitted purely as a **short-circuit
optimization** to avoid a wasted inference call on an obvious retry. It is explicitly *not* the
correctness mechanism: between that read and the write, a concurrent request can insert the same
`event_id`, and only the unique constraint closes that race.

**Why not `event_id` as `_id`?** Overloading the storage primary key with a domain identifier
couples the domain contract to one engine's identity model, and pushes a MongoDB-specific concept
into a boundary that is supposed to be storage-agnostic. Separating them costs one index and buys
portability.

**Portability.** A PostgreSQL implementation stores `event_id` as a column with
`UNIQUE (event_id)` alongside whatever surrogate primary key its schema prefers, maps its
unique-violation error to the same `DuplicateEventId` outcome, and reproduces the behavior table
above exactly — with no MongoDB identity concept appearing anywhere in the application layer.

**The mechanism is deliberately not implemented in this phase.** The contract, the response
semantics, and the uniqueness constraint are fixed now so that implementation is mechanical later.

### Consequences

* `event_id` must be opaque to the server: no parsing, no ordering assumptions.
* One additional unique index is maintained on writes — the deliberate cost of keeping domain and
  storage identity separate.
* Because idempotency lives in `IngestTelemetry` and the repository rather than in the HTTP layer,
  a future MQTT adapter inherits it with no new logic.
* `TelemetryRepository` exposes `find_by_event_id` and a `save` that signals `DuplicateEventId`; no
  storage-assigned identity appears in the port
  ([ADR-0008](#adr-0008-deliberately-limited-ports)).
* Retention, archival, and any future migration key on `event_id`, which is stable across storage
  engines, rather than on an engine-specific identifier.
* On a payload conflict the *write* is rejected by the constraint, but the *request* is not: it
  resolves as first-write-wins and returns `200 OK` with the stored original, logging the conflict
  rather than surfacing a `409`. The original event is preserved either way, and stricter handling
  waits for evidence that conflicts occur.

---

## ADR-0006: Fail-open persistence when inference fails

**Status:** Accepted · 2026-08-31

### Context

Ingestion calls the Inference Service synchronously ([ADR-0002](#adr-0002-synchronous-http-inference-for-the-mvp)).
That service can time out, return an error, fail to load its artifact, or be unreachable during a
deploy. The ingest path must decide what happens to the telemetry when that occurs.

### Decision

The two downstream dependencies get **opposite** policies, and the terms are used in exactly this
sense throughout the documents:

| Dependency fails | Policy |
| --- | --- |
| **Inference failure** | **fail-open for telemetry persistence** — the event is still stored, ingestion succeeds |
| **Persistence failure** | **fail-closed for ingestion** — ingestion fails with `503`, nothing is acknowledged that was not stored |

**Inference failure → fail-open for telemetry persistence.** When inference fails, the Telemetry
Service:

1. **preserves the telemetry** — the event is stored with its canonical metrics;
2. **records `inference.status = FAILED`** with an `error_code` naming the failure class;
3. **invents nothing** — `is_anomaly` and `anomaly_score` remain `null`;
4. **exposes the incompleteness** — the ingest response and telemetry queries show `FAILED`, and
   the anomalies endpoint excludes such events entirely.

**Persistence failure → fail-closed for ingestion.** If the event cannot be stored, the request
returns `503 Service Unavailable` rather than a success code, so the client's retry — safe by
[ADR-0005](#adr-0005-event_id-as-the-idempotency-key) — is the recovery mechanism. Fail-open
applies to the derived score, never to the measurement.

A duplicate `event_id` is **not** a persistence failure: it is a successful, expected outcome of
the uniqueness constraint and returns `200 OK` with `duplicate: true`.

### Rationale

Telemetry is the irreplaceable asset; an anomaly score is derived data that can be recomputed.
Losing a measurement because a model server was restarting is the worst available outcome.

Defaulting a failed inference to `is_anomaly: false` would record a false negative as if the model
had spoken — the same shape of error as a missed fault. An unscored event is not a non-anomaly, so
the status is recorded rather than guessed, and the anomalies endpoint never returns it.

### Consequences

* Stored events have three meaningful states: scored-normal, scored-anomalous, and unscored.
  Every consumer must handle the third.
* Anomaly queries can under-report during an inference outage. That is visible and correct, rather
  than invisible and wrong.
* `inference.status` makes recovery straightforward: a backfill job can re-score `FAILED` events
  later, and a circuit breaker can be added without changing the stored shape.
* Clients receive `201` on a successful ingest with failed inference, so they must read
  `inference.status` rather than the HTTP code to know whether a score exists.

---

## ADR-0007: Transport-independent ingestion use case

**Status:** Accepted · 2026-08-31

### Context

REST is the MVP's only ingestion transport. MQTT is an anticipated addition. The common failure
mode is that ingestion logic — validation, normalization, skew checks, idempotency, inference,
persistence — accretes inside HTTP handlers and must then be duplicated, and subsequently
diverges, for the second transport.

### Decision

Telemetry ingestion is a single application use case, **`IngestTelemetry`**, which owns all of that
logic and accepts a **domain command**, not an HTTP request.

* The FastAPI router is a **transport adapter**: parse, validate shape, map to the command, map the
  result and errors back to HTTP.
* No business logic in routers; no framework types in the application or domain layers.
* Dependency direction is `api → application → domain`, with `infrastructure` implementing
  application ports.
* A future MQTT adapter calls the same use case; only transport concerns (subscription, QoS,
  back-pressure) are new. `ingest.transport` records the source on the stored document.

### Consequences

* One extra mapping layer between HTTP schemas and the command — the deliberate cost.
* Ingestion behavior is testable without a web server or a broker.
* Adding a transport cannot silently change ingestion semantics, because there is only one place
  where those semantics live.
* Transport-specific concerns must stay in adapters; leaking one into the use case would defeat
  the decision.

### Related

This decision is what makes extension points 1–4 in
[ARCHITECTURE.md §14](ARCHITECTURE.md#14-future-extension-points) additive rather than rewrites.

---

## ADR-0008: Deliberately limited ports

**Status:** Accepted · 2026-08-31

### Context

The design goal requires clean extension points *and* a system small enough to implement and
explain. Ports-and-adapters, applied uniformly, produces an interface for everything and obscures a
small system with indirection. Applied nowhere, it welds the application to its infrastructure.

### Decision

Create exactly **two** ports in the MVP:

| Port | Justification | MVP adapter |
| --- | --- | --- |
| `InferencePort` | External networked system with independent failure modes; async and multi-model alternates are anticipated | `HttpInferenceClient` |
| `TelemetryRepository` | External infrastructure; PostgreSQL + JSONB is a documented viable alternative, and tests need a fake | `MongoTelemetryRepository` |

`TelemetryRepository` is **specific** — `save`, `find_by_event_id`,
`find_by_vehicle_and_time_range`, `find_anomalies_by_vehicle_and_time_range` — shaped by actual
access patterns, not a generic `Repository[T]`. It speaks **domain identity only**: `save` signals
a uniqueness violation as a `DuplicateEventId` outcome, and no storage-assigned identity crosses
the boundary ([ADR-0005](#adr-0005-event_id-as-the-idempotency-key)).

Deliberately **not** abstracted: `TelemetryNormalizer` (pure domain logic), the clock (a callable
injected for tests, not a port), FastAPI/Pydantic (the framework *is* the adapter), configuration,
logging, and any event bus, unit-of-work, DI container, plugin registry, or model registry.

There is no hexagonal-architecture framework: no base classes, no container, no registration
machinery. Ports are plain Python protocols/ABCs.

### Rationale

An abstraction is earned by a second real implementation or by an external system with independent
failure modes. Both ports meet that bar; nothing else in the MVP does. Speculative interfaces cost
indirection now and rarely fit the requirement that eventually arrives.

### Consequences

* Concrete types are used directly in most of the codebase, which keeps it readable.
* Adding a third port later is a small, local refactor — extract the interface when the second
  implementation exists.
* The bar is explicit, so "should this be an interface?" has a stated answer rather than a
  per-author preference.

### Revisit when

A second real implementation appears for a currently-concrete component, or a component grows an
independent external failure mode.

---

## ADR-0009: `PENDING` as the unscored inference state

**Status:** Accepted · 2026-08-31

### Context

Persistence was implemented before inference. That produces a state the original
design never modelled: an event that is stored but has never been scored. The
`inference.status` vocabulary held only `COMPLETED` and `FAILED`, because the approved write path
calls the model *before* the single persistence write, so every stored event already had a verdict.

Neither existing value is true of an unscored event. `COMPLETED` would fabricate a verdict.
`FAILED` would claim the model was called and did not answer, when it was never called at all —
and that distinction matters operationally, because a genuine `FAILED` is a signal to investigate
the inference service.

### Decision

Add a third state, **`PENDING`**: the event is persisted and has not been scored.

* `PENDING` sets `is_anomaly`, `anomaly_score`, `model_name`, `model_version` and `error_code` to
  null. Only `COMPLETED` carries a verdict.
* A `PENDING` event is **never** returned by the anomalies endpoint, and is excluded from the
  partial anomaly index. An unscored event is not a non-anomalous event.
* `PENDING` describes a state, not an activity. It does **not** mean background scoring is running
  or scheduled; there is no queue, worker or background task, and none is introduced by this
  decision.
* Once synchronous inference is integrated, a newly ingested event is written with `COMPLETED` or
  `FAILED` directly, and `PENDING` is what a record keeps only until something scores it.

### Consequences

* Ingestion is a complete, honest operation before inference exists: `201 Created` with a truthful
  state rather than a fabricated verdict or a misleading `501`.
* Every consumer must handle three states, and must not read "not `COMPLETED`" as "normal".
* The vocabulary already matches the asynchronous shape described in
  [ADR-0002](#adr-0002-synchronous-http-inference-for-the-mvp), so adopting a queue later needs no
  further status change.

### Alternatives considered

* **Reuse `FAILED`** — untrue, and it would hide real inference failures among events that were
  never scored.
* **Keep returning `501` until inference exists** — the event *was* stored, so the response would
  misreport what happened, and a client would retry indefinitely against a successful write.

---

## ADR-0010: Train the model during the image build

**Status:** Accepted · 2026-09-01

### Context

The repository ignores `*.joblib`, so the ~20 MB model artifact is not committed. That is the right
call for a generated binary, but it leaves a gap once the system is containerized: the Inference
Service refuses to serve without an artifact, and a developer cloning the repository has none.

Something has to produce it, and the options differ mainly in *where* they put that work: commit the
binary, train when the container starts, train on the first request, or train while building the
image.

### Decision

The Inference Service image runs `ml/train.py` as a build step. The artifact is baked into the
image; the container loads it once at startup and never trains.

```text
source code → docker build → ml/train.py → model.joblib in the image
            → container start → artifact loaded once → predictions served
```

This holds the line already drawn between training and serving
([§11](ARCHITECTURE.md#11-model-and-inference-design)): the running service still never trains, not
at startup and not in a request handler. Training simply moved from a developer's shell into the
build.

It works because training is deterministic — fixed seeds, explicit hyperparameters, no network and
no external dataset — so the same source produces the same model, and an image is reproducible from
the commit it was built from. Nothing about the model itself changes: same feature order,
hyperparameters, score orientation, name, version and artifact schema.

### Consequences

* `docker compose up --build` works from a fresh clone with no manual preparation.
* The image is self-contained: no artifact volume, no download at startup, no registry to reach.
* Image builds are slower by the training time — under a second here — and the image carries the
  artifact's size.
* Retraining means rebuilding: `docker compose build --no-cache`. In a container deployment the
  image tag becomes part of the model's identity, alongside `model_version`.
* Running outside Docker still needs `python ml/train.py` once, exactly as before.

### Alternatives considered

* **Commit the artifact.** Puts a large regenerable binary into Git history, where it would drift
  from the training code silently. It also contradicts the existing `.gitignore` policy.
* **Train at container startup.** Erases the training/serving separation, makes startup slow and
  variable, and retrains on every restart and every replica.
* **Train on the first request.** Worse still: it puts training inside a request handler, which
  [§11](ARCHITECTURE.md#11-model-and-inference-design) explicitly forbids.
* **Mount an artifact from the host.** Reintroduces the manual step containerization exists to
  remove, and makes the image depend on state outside it.

### Revisit when

A model registry exists, or models are trained on real data — at which point the artifact becomes an
*input* to the build rather than a product of it, and this decision is replaced rather than amended.

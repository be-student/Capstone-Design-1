# F2 — Page 15 System Health

Scope: `src/dashboard/system_health_view.py` only. All other dashboard files
are owned by sibling F-agents and were left untouched.

## Reasoning

### Defect 1 — same-page contradiction A (Experiments=0 vs Total Runs=3)

Iter9 audit a4 P15: *"MLflow KPI says Experiments=0 but 'Total Runs=3' —
same page, two MLflow facts in mutual contradiction."* Root cause: two
independent code paths. The Experiments KPI in `_render_service_cards`
read `len(svc.get("experiments", []))` from `check_mlflow_health`, while
the Run-History KPI read `len(data_loader.load_mlflow_runs())`. When the
live MLflow API returned no experiments but the cached snapshot still had
runs, the page printed `0` and `3` simultaneously.

Fix shape: load `mlflow_runs_df` once at the top of `render_system_health`,
pass it down into `_render_service_cards` and `_render_mlflow_tracking`,
and reconcile in the card by taking
`max(len(live_experiments), distinct experiments inferred from runs)`.
When the cached snapshot has runs but no `experiment_id` column, fall
back to `1` so the invariant *"runs > 0 implies experiments >= 1"*
always holds. The Total-Runs KPI is now also rendered on the card so
the same number is visible in both places.

### Defect 2 — same-page contradiction B (Operational vs Drift=RED)

Iter9 audit a4 P15: *"All Systems Operational ✅ while Drift Status =
RED"*. Root cause: `get_system_health_summary` only aggregated Redis,
MLflow, and Pipeline. The drift subsystem was rendered separately and
never folded into the rollup, so the headline could not see it.

Fix shape: extend `get_system_health_summary` to accept an optional
`data_loader`, query `load_drift_history()`, map `green/yellow/red` to
`STATUS_HEALTHY/DEGRADED/DOWN`, add a `drift` child, and propagate the
worst-child state (any DOWN ⇒ DOWN, any DEGRADED ⇒ DEGRADED). The
banner text was also updated: STATUS_DEGRADED now reads
*"Degraded — Investigate Subsystems"* and the headline is rendered with
`st.warning` / `st.error` so the icon, colour, and copy can no longer
disagree. A `Non-healthy subsystems: …` caption surfaces exactly which
child dragged the rollup down.

### Defect 3 — Page 14 ↔ Page 15 MLflow contradiction

Iter9 audit a4: *"Page 14 says MLflow tracking server **not available**;
Page 15 says **Connected to MLflow tracking server**."* Root cause:
`check_mlflow_health` short-circuited on `sqlite://` — it returned
`status=HEALTHY, connected=True` whenever the sqlite file *existed on
disk*, with no actual API probe. Page 14 uses a different (live)
mlflow client probe and so honestly admitted it was unreachable.

Fix shape: rewrite `check_mlflow_health` so connection is decided by
the same canonical path Page 14 uses — `mlflow.search_experiments()`
must succeed AND return at least one experiment. The sqlite-file-exists
check is now a precondition, not a positive result. Added a new
`mlflow_status_banner(mlflow_health)` helper that returns
`(level, message)` for the banner, and `_render_mlflow_tracking` now
sources its banner copy from that helper. A future Page 14 fix can
import the same helper for guaranteed agreement; for the time being
the warning copy mirrors Page 14's *"MLflow tracking server not
available — showing cached experiment data from artifacts."*

### Defect 4 — hollow service health

Iter9 audit a4 P15: *"All-green checkmarks on services that have nothing
inside them (0 streams, 0 experiments)."* "Healthy" reduced to "TCP
socket opened" is misleading.

Fix shape: keep service-level `status=HEALTHY` (the probe really did
succeed) but qualify the `Connected` metric label.
- Redis: when `connected and sum(stream_lengths)==0` ⇒ label reads
  `"Yes (idle — no traffic)"`.
- MLflow: when `connected and runs_total==0` ⇒ `"Yes (idle — no runs
  logged)"`. When `not connected and runs_total>0` ⇒ `"No (showing
  cached runs)"`.
- Pipeline: when `artifacts==0` and no models ⇒ idle caption.

### Defect 5 — time anchor split

Iter9 audit a4 P15: *"Throughput chart dated Oct 15 2024 while drift
chart dated May 10 2026 — same page."* The cached `scoring_throughput`
fixture is anchored to Oct-2024, so the 24-hour line chart was always
~19 months out of phase with the live drift charts.

Fix shape: added `_throughput_freshness(df)` and a new constant
`THROUGHPUT_FRESHNESS_HOURS = 48`. Before rendering the chart, compute
the age of the latest `timestamp`; if the data is older than 48 h, the
chart is suppressed and replaced by an `st.warning` that explicitly
states the staleness and points the operator at restarting the scoring
pipeline. When the data is fresh, the section header changes from a
hard-coded *"(24h)"* to *"(last 24h)"* and a `Last refresh: <ISO>`
caption is added. The Avg Throughput / Latency / Error Rate KPIs are
also moved inside the freshness branch so they are not surfaced on
top of suppressed data.

## Changes (file:line:before → after)

All edits in `src/dashboard/system_health_view.py`.

- L13–18 imports: added `timedelta, timezone` (only `datetime` was
  imported), added `Tuple` to the typing import. Required for the new
  freshness helper signature.
- L25–46: defensive imports of `format_count` and `drift_trend_guard`
  from `src.dashboard.utils.dashboard_helpers` (per the brief, F1 may
  not have committed yet). Inline fallbacks preserve behaviour.
  Added `THROUGHPUT_FRESHNESS_HOURS = 48` constant.
- `check_mlflow_health` (~L155 onward): removed the
  *"sqlite file exists ⇒ HEALTHY"* short-circuit. Connection is now
  decided by `mlflow.search_experiments()` returning ≥1 experiment.
  Empty registry ⇒ `STATUS_DEGRADED, connected=False`. Exceptions ⇒
  `STATUS_DOWN`. Added new `mlflow_status_banner(...)` helper alongside.
- `_drift_alert_to_status` (new, ~L380): maps drift alert level
  (green/yellow/red) ⇒ service status.
- `get_system_health_summary` (~L395): signature gained
  `data_loader: Any = None` (backward compatible — existing callers
  passing only `config` continue to work). Now also queries
  `load_drift_history()` defensively, includes a `drift` child in
  `services`, and propagates worst-child status (any DOWN ⇒ DOWN,
  any DEGRADED ⇒ DEGRADED, otherwise HEALTHY).
- `render_system_health` (~L488): pre-loads
  `mlflow_runs_df = data_loader.load_mlflow_runs()` once, passes it
  into `_render_service_cards` and `_render_mlflow_tracking`. Passes
  the `data_loader` into `get_system_health_summary` so drift folds
  into the rollup.
- `_throughput_freshness` (new, ~L541): computes
  `(latest_timestamp, age_hours)` from the throughput frame.
- `_render_overall_health` (~L570): banner now rendered with
  `st.warning` / `st.error` for non-healthy states; added
  `Non-healthy subsystems: …` caption; STATUS_DEGRADED copy changed
  from *"Some Services Degraded"* to *"Degraded — Investigate
  Subsystems"*; service-count text uses `format_count`.
- `_render_service_cards` (~L627): new optional `mlflow_runs_df`
  param. Redis card: `Connected` label switches to
  `"Yes (idle — no traffic)"` when stream totals are zero; stream
  lengths use `format_count`. MLflow card: reconciles
  `Experiments` against the runs DataFrame
  (`max(live, distinct from runs, …)`); adds a `Total Runs` metric
  on the same card so both halves of the page agree by construction;
  qualifies the `Connected` label for idle / cached-only states.
  Pipeline card: uses `format_count` and surfaces an "idle" caption
  when artifacts and models are both empty.
- `_render_streaming_status` throughput section (~L765): chart
  rendering wrapped in a freshness check. Stale data ⇒ no chart,
  warning banner with the last sample's ISO timestamp. Fresh data ⇒
  header now reads *"Scoring Throughput (last 24h)"* with a
  `Last refresh:` caption above the chart.
- `_render_mlflow_tracking` (~L805): new optional `mlflow_runs_df`
  parameter (reuses the cached frame). Banner copy now sourced from
  `mlflow_status_banner(...)` so Page 14 and Page 15 cannot drift.
  The "no experiments" info copy now explicitly mentions that the
  run history is loaded from cached artifacts.

## Public API preservation

- `resolve_redis_connection_config(config)` — unchanged.
- `check_redis_health(config)` — unchanged.
- `check_mlflow_health(config)` — same signature, same return-dict
  schema (`status, connected, tracking_uri, experiment_name,
  experiments, total_runs, recent_runs, best_run, error`). Behaviour
  for sqlite tracking URIs is now stricter (file-exists is no longer
  enough).
- `check_pipeline_health(config)` — unchanged.
- `get_system_health_summary(config)` — backward compatible:
  `data_loader` is keyword-only with default `None`. `services` dict
  now has an extra `drift` key in addition to `redis / mlflow /
  pipeline`; the existing keys and their schemas are unchanged.
- `render_system_health(st_module, config, data_loader=None)` — unchanged.
- `mlflow_status_banner(mlflow_health)` — new public helper, intended
  for Page 14 to import once F-agent for `app.py` is ready.

## Closes

- a4 P15 contradiction A (Experiments=0 vs Total Runs=3) — single
  cached `mlflow_runs_df` reconciles both KPIs.
- a4 P15 contradiction B (All Systems Operational vs Drift=RED) —
  drift folded into the rollup with worst-child propagation.
- a4 P14 ↔ P15 MLflow status contradiction (from the P15 side) —
  `check_mlflow_health` now requires a successful API probe;
  `mlflow_status_banner` provides shared banner copy.
- a4 hollow-health greens — `Connected` label qualified with
  *"(idle — …)"* / *"(showing cached runs)"* when activity is zero.
- a4 time anchor split — Oct-2024 throughput fixture suppressed when
  older than 48 h, replaced by a staleness warning; fresh data gets a
  `Last refresh` timestamp and a *"last 24h"* (not hard-coded
  *"(24h)"*) heading.

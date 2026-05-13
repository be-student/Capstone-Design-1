# iter13 G3 — `src/dashboard/app.py` override removals

Owner: FIX AGENT G3 (iter13)
Scope: `src/dashboard/app.py` (only). Other modules owned by G1/G2/G4.

## Inputs

- `_test_results/iter12/audit_B_lineage.md` §Risks #1-2: Page 02 KPIs
  derived from a hardcoded `[[350,50],[80,120]]` confusion-matrix fixture.
- `_test_results/iter12/audit_C_kpi_sources.md` Top fishy KPIs #1-5.
- `src/dashboard/data_loader.py` already exposes `DashboardArtifact`
  (`@dataclass` with `data`, `is_real`, `source_path`, `reason`, `name`,
  `extra`) and `as_artifact=True` on `load_survival_data`,
  `load_confusion_matrices`, `load_roc_data`, `load_survival_curves`,
  `load_scoring_history`, `load_scoring_throughput`,
  `load_retention_offers`. `load_drift_history` does NOT yet accept
  `as_artifact` — handled by the defensive helper.

## Changes (all in `src/dashboard/app.py`)

### 1. Defensive imports + `_load_as_artifact` helper (new)

Added after the `_probe_mlflow_status` import block:

- `try: from src.dashboard.data_loader import DashboardArtifact / except
  ImportError: DashboardArtifact = None` so older G2 builds do not break
  the import.
- `def _load_as_artifact(data_loader, method_name, *args, **kwargs)`
  returns `(payload, is_real, missing_reason)`. It calls
  `method(*args, as_artifact=True, **kwargs)`; on `TypeError` it falls
  back to the legacy call signature and marks the result as not-real
  (so callers fall through to the explicit error branch). It also
  duck-types so any object exposing `.is_real` / `.data` / `.reason` is
  understood, future-proofing against class renames.

### 2. P0 — `render_model_performance` (lines ~406-700)

Audit cite: `audit_B_lineage.md` §Risks #1, `audit_C_kpi_sources.md`
"Top fishy KPIs #1".

- **Removed** the override block at the previous `app.py:439-463` that
  rewrote `metrics[<model>]["precision" / "recall" / "accuracy" /
  "f1_score"]` from a `[[350,50],[80,120]]` fixture matrix.
  Headline P/R/F1/Accuracy now come directly from
  `model_metrics.json` via `load_model_metrics()`.
- Confusion-matrix tile rendering is now gated on
  `_load_as_artifact(data_loader, "load_confusion_matrices")`. If
  `is_real=False`, render
  `st.error("Real confusion-matrix data missing — run `python -m
  src.main --mode all` ...")` instead of plotting fixture tiles.
- The test-set-size caption is recomputed from the real matrix when
  available (previously it was derived from the fixture sum 600,
  which contradicted the real `model_metrics.json.test_size = 3334`).

### 3. P1 — `render_survival_analysis` (lines ~1859-2050)

Audit cite: `audit_B_lineage.md` §Risks #5, `audit_C_kpi_sources.md`
"Top fishy KPIs #2".

- `load_survival_data()` now called via `_load_as_artifact`. When
  `is_real=False`, render `st.error("Real survival artifacts missing — run
  `python -m src.main --mode all`...")` and `return`. The previous code
  silently rendered KPIs derived from
  `_load_survival_from_segments` (`duration_days = 365 × (1 -
  churn_probability)`).
- `load_survival_curves()` likewise gated; renders an error and returns
  before any Kaplan-Meier / hazard / event-rate / duration-distribution
  chart is drawn.

### 4. P1 — `_render_scoring_status_tab` (Page 13 tab a)

Audit cite: `audit_B_lineage.md` §Risks #4, `audit_C_kpi_sources.md`
"Top fishy KPIs #3, #5".

- `load_scoring_throughput()` gated via `_load_as_artifact`. When
  `is_real=False`, render
  `st.warning("Throughput telemetry not yet wired to Redis stream.
  Connect real consumer-group counters for production.")` and hide the
  throughput + latency charts.
- `load_scoring_history()` gated via `_load_as_artifact`. When
  `is_real=False`, render `st.error("Real scoring-history data missing
  ...")` and set the caption to `Data window: — · Last refresh: — ·
  <model_stamp>`. KPI strip + downstream charts (distribution, risk
  bars, detailed table) skipped.

### 5. P1 — `_render_retention_offers_tab` (Page 13 tab b)

Audit cite: `audit_C_kpi_sources.md` "Top fishy KPIs #4".

- `load_retention_offers()` gated via `_load_as_artifact`. When
  `is_real=False`, render `st.error("Real retention-offer data missing —
  run the retention optimizer ...")`, caption to `Last refresh: —`, and
  early-return.
- The `Total Offers` denominator that previously called
  `load_scoring_history()` directly now also goes through
  `_load_as_artifact` and counts 0 when the scoring-history artifact is
  not real, so the `N / scored` ratio is never derived from synthetic
  data.

### 6. P1 — `_render_monitoring_tab` (Page 13 tab c)

Audit cite: `audit_C_kpi_sources.md` Page 13c, `audit_B_lineage.md`
flow 5 (`load_drift_history` derives a 1-row frame from
`monitoring_report.json` when `drift_history.csv` is absent).

- `load_drift_history()` and `load_scoring_history()` both invoked via
  `_load_as_artifact`. `load_drift_history` does not yet accept
  `as_artifact=True` — the defensive helper catches the `TypeError`,
  returns the legacy payload tagged not-real, and the existing
  `drift_trend_guard` is augmented to also short-circuit when the
  artifact is not real, surfacing `"Insufficient drift history — run
  pipeline to populate drift_history.csv"`. The single-row case
  continues to show the latest-alert-level KPI (still backed by a real
  `monitoring_report.json`).
- The "Scoring Quality Metrics" subsection now refuses to plot mean-
  churn-probability / scoring-volume charts when `scoring_history` is
  not a real artifact, replacing them with an
  `st.info(...)` banner.

### 7. P1 — `render_mlflow_experiments` (Page 14)

Audit cite: `audit_B_lineage.md` Risk #8.

- When `_probe_mlflow_status` reports `connected=True`, the page now
  queries `mlflow.tracking.MlflowClient.search_runs(...)` against the
  configured `tracking_uri` / `experiment_name` and renders the live
  run list (with `st.success("Live MLflow query — N runs from tracking
  server.")`).
- When MLflow is unreachable or the live query is empty, the page falls
  back to `data_loader.load_mlflow_runs()` (cached
  `model_performance_history.csv`) and prepends
  `st.info("Cached snapshot — N=X runs from
  `results/model_performance_history.csv`. The tracking server was not
  reachable, so this is not the live run list.")`.
- KPI cards get help-tooltips clarifying the source. "Total Training
  Time" tooltip explicitly notes the cached snapshot fills missing
  `training_time_s` with a 1.0 s default.

## Validation

- `python -c "import ast; ast.parse(open('src/dashboard/app.py',
  encoding='utf-8').read())"` → exits 0 (syntax OK).
- Did NOT touch `src/dashboard/data_loader.py`,
  `src/dashboard/monitoring_view.py`,
  `src/dashboard/system_health_view.py`,
  `src/dashboard/recommendations_view.py`, `src/main.py`, or pipeline
  modules — those belong to other fix agents (G1/G2/G4).
- Behavior when G2's `DashboardArtifact` is missing on import:
  `DashboardArtifact = None`, `_load_as_artifact` falls back to legacy
  call signature and marks every loader output as not-real. Pages
  therefore display the "real artifact missing" error universally
  — which is the safe direction per the brief ("explicit `st.error`
  rather than silent fixture").

## Files modified

- `src/dashboard/app.py`

## Files NOT modified (other agents)

- `src/dashboard/data_loader.py` (G2)
- `src/dashboard/monitoring_view.py` (G4)
- `src/dashboard/system_health_view.py` (G4)
- `src/dashboard/recommendations_view.py` (G4)
- `src/main.py` (G1)

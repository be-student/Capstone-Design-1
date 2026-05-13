# iter13 G4 — Page 08 / 09 / 15 view fix log

Agent: FIX AGENT G4
Scope: `src/dashboard/monitoring_view.py`, `src/dashboard/system_health_view.py`,
`src/dashboard/recommendations_view.py`.
Inputs read: `_test_results/iter12/audit_C_kpi_sources.md` (Page 08, 09, 15
fixture KPIs).

## Goal

Honor G2's forthcoming `DashboardArtifact` (`is_real`, `reason`, `data`)
return type from `load_scoring_throughput`, `load_drift_history`,
`load_retention_offers`, `load_mlflow_runs`. When the loader reports an
artifact is NOT real, surface a Streamlit `error`/`warning` instead of
silently rendering a synthetic chart.

## Defensive contract

G2's loader signature may not have shipped yet. All three view modules now
share a small adapter:

```python
try:
    from src.dashboard.data_loader import DashboardArtifact  # type: ignore
except Exception:
    DashboardArtifact = None  # type: ignore

def _load_artifact_safely(loader_callable, *args, **kwargs):
    if loader_callable is None:
        return None, None
    try:
        result = loader_callable(*args, as_artifact=True, **kwargs)
    except TypeError:
        # Old signature — caller does not accept as_artifact.
        try:
            payload = loader_callable(*args, **kwargs)
        except Exception:
            return None, None
        return payload, None
    except Exception:
        return None, None
    payload = getattr(result, "data", result)
    return payload, result

def _artifact_marked_unreal(artifact) -> bool:
    if artifact is None:
        return False  # legacy fallback — preserve old behavior
    is_real = getattr(artifact, "is_real", None)
    if is_real is None:
        return False
    return bool(is_real) is False
```

- `TypeError` on `as_artifact=True` is caught and the old positional call
  is retried so the page keeps rendering when G2 has not yet landed.
- A loader returning `None` (e.g. legacy path raised) is treated as
  "no artifact info available" — we do NOT trigger the new warning to
  avoid false-positive callouts while G2 is being merged.
- The artifact wrapper is duck-typed: `.data` (payload), `.is_real`,
  `.reason`. No hard import of `DashboardArtifact` is required for the
  views to function.

## Per-file changes

### `monitoring_view.py` (Page 08)

1. New helpers at module scope:
   - `_load_artifact_safely(loader_callable, ...)` — see above.
   - `_artifact_marked_unreal(artifact)` — explicit `is_real=False` check.
   - `_render_artifact_reason(st, artifact)` — small `st.caption` writing
     `Reason: <artifact.reason>` when present.
2. `render_model_monitoring`:
   - `drift_history` and `scoring_throughput` are now loaded via
     `_load_artifact_safely`, capturing `(payload, artifact)` for each.
   - When `_artifact_marked_unreal(drift_artifact)` is True the section
     short-circuits to
     `st.error("Real drift history missing — run the pipeline to populate
     results/drift_history.csv (or monitoring_report.json).")` and skips
     `_render_drift_section`.
   - The existing `drift_trend_guard` (n<5 case) is left intact inside
     `_render_drift_section`. The `is_real` check is **separate**:
     artifact-existence vs sufficient-history. Either failure mode can
     fire independently.
3. `_render_throughput_section(st, scoring_throughput, artifact=None)`:
   - New `artifact` param (default `None` — legacy callers unaffected).
   - When `_artifact_marked_unreal(artifact)` is True, render
     `st.error("Real scoring throughput missing — run pipeline to populate
     results/scoring_throughput.csv.")` and return early. None of the
     throughput / latency / error-rate charts are drawn, nor are the
     fixture-derived KPI tiles (49 req/min, 19.1 ms, 1.03%) that the
     iter12 audit flagged for Page 08.

### `system_health_view.py` (Page 15)

1. Same defensive helpers added (`_load_artifact_safely`,
   `_artifact_marked_unreal`, `_artifact_reason`).
2. `render_system_health`:
   - `data_loader.load_mlflow_runs()` is replaced with
     `_load_artifact_safely(data_loader.load_mlflow_runs)`, yielding
     `(mlflow_runs_df, mlflow_runs_artifact)`. The latter is threaded into
     `_render_service_cards` as a new keyword arg.
3. `_render_service_cards(..., mlflow_runs_artifact=None)`:
   - Inside the `mlflow` card branch, after the existing
     `Connected / Experiments / Total Runs` metrics, append
     `st.caption(f"Experiments cached: {format_count(runs_total)} rows")`
     when `_artifact_marked_unreal(mlflow_runs_artifact)` AND
     `runs_total > 0`. This mirrors the requirement
     ("Experiments cached: N rows" when `is_real=False, reason="cached
     fallback"`).
   - If `artifact.reason` is non-empty, a follow-on caption
     `Reason: <reason>` is shown so the operator can see *why* the loader
     returned cached data. Connection status copy ("Connected: No /
     showing cached runs") was already handled in iter11 and is left
     unchanged.
4. `_render_streaming_status`:
   - Throughput tile now uses `_load_artifact_safely` on
     `load_scoring_throughput`.
   - When the artifact is marked unreal, the chart and the
     freshness/stale banner are both suppressed and replaced with
     `st.error("Real scoring throughput missing — run pipeline to populate
     results/scoring_throughput.csv.")`. The legacy "stale telemetry"
     warning continues to fire when the artifact IS real but the latest
     sample is older than `THROUGHPUT_FRESHNESS_HOURS = 48`. The two
     failure modes are handled independently.

### `recommendations_view.py` (Page 09)

1. Module-level defensive helpers (`_load_artifact_safely`,
   `_artifact_marked_unreal`).
2. `render_recommendations_view`:
   - `data_loader.load_retention_offers()` replaced with
     `_load_artifact_safely(data_loader.load_retention_offers)`. The
     existing legacy fallback to `_generate_sample_retention_offers`
     inside the loader is left in place; this fix only changes how the
     view *reacts* to a sample-fallback return.
   - Section 5 (Cost-Benefit) gating:
     - If `_artifact_marked_unreal(retention_offers_artifact)` is True →
       render `st.warning("Retention offer breakdown not yet computed —
       top KPIs above show full population stats from real
       recommendations.csv.")` and **do NOT** call
       `_render_cost_benefit_analysis`. This closes the iter12 audit
       finding that the Cost-Benefit strip (Total Campaign Cost ₩1.21M,
       Est. Revenue Saved ₩10.89M, ROI 9.0x, Avg Treated Uplift 10.88%)
       was sourced from `_generate_sample_retention_offers`.
     - Legacy `if not retention_offers.empty` branch is preserved as the
       `elif` clause so an artifact-aware loader returning a real
       non-empty frame keeps rendering the existing cards.

## Constraints honored

- DashboardArtifact is imported defensively inside a `try/except` block.
  None of the three modules raise at import time if G2 has not yet shipped
  the new type.
- `as_artifact=True` is attempted with `_load_artifact_safely`. The
  resulting `TypeError` on the old loader signature falls back to the
  legacy positional call so existing behavior is unchanged.
- No new features beyond the three iter12 audit findings:
  - Page 08 throughput KPIs (49 / 19.1 / 1.03%)
  - Page 09 cost-benefit strip (₩1.21M / ₩10.89M / 9.0x)
  - Page 15 throughput tile + MLflow cached-runs disclosure
- No edits outside the three permitted files. Pipeline, data_loader, and
  app.py were not touched.

## Verification

- All three files re-parse cleanly under `ast.parse(...)`
  (`python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in [...]]; print('OK')"`
  returns `OK`).
- No new third-party imports. `DashboardArtifact` import is guarded.
- Helpers are module-scope `def`s; reference resolution at call time keeps
  ordering insensitive.

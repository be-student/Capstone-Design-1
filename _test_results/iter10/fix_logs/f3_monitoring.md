# F3 — Monitoring View Fix Log (Page 08)

**Agent:** F3 (parallel remediation)
**Files modified:** `src/dashboard/monitoring_view.py` (only)
**Sources:** `_test_results/iter9/iter9_audit_a4.md`, `_test_results/iter9/page_data/08_model_monitoring.md`

---

## Reasoning

The iter9 audit identified six page-08 defects, all stemming from one structural problem: monitoring_view.py was rendering visuals that overstated what the underlying data could actually support. A 1-row drift history was drawn as a 1.5 ms "trend"; three sequential function calls were drawn as a 0.1 ms "training timeline"; an Oct-2024 throughput fixture sat on the same page as a May-2026 drift snapshot with no freshness label; a green "no degradation" success banner sat next to a RED drift KPI; a 1.03% error rate had no SLO callout despite being 10× the standard SaaS target; and Page-07 Kaplan–Meier curves were being re-rendered here, leaking scope.

The fix is conservative: gate every "over time" chart through `drift_trend_guard` (the F1 helper) and degrade gracefully to a single-snapshot panel when the series is too sparse; derive the page headline banner from drift status (the authoritative health signal); add `Last update:` captions to every chart so the time-anchor split is no longer hidden; surface the SaaS error-rate SLO breach as a red callout; remove the duplicated survival section.

I did **not** mutate any file outside `monitoring_view.py`. The F1 helper import is wrapped in `try/except` with a local fallback so the page never crashes if the helper is not yet available.

---

## Changes

### Imports / module preamble
- Added guarded `from src.dashboard.utils.dashboard_helpers import drift_trend_guard` with a local fallback that mirrors the helper contract `(ok, message)`. Keeps the page resilient to F1 ordering.
- Added `from datetime import datetime` for freshness comparison.
- Added module-level constant `ERROR_RATE_SLO_TARGET = 0.001` (0.1% SaaS SLO target).
- Updated module docstring to remove the "Survival Analysis" claim — that page now only owns monitoring.

### `render_model_monitoring`
- Page header changed from "Model Monitoring & Survival Analysis" → "Model Monitoring".
- Removed `survival_curves` / `survival_data` data-loader calls (they are no longer rendered here).
- Removed the Section 4 "Survival Curves Quick Reference" block; renumbered "Monitoring Configuration" to Section 4.
- Passes `performance_alerts` into `_render_drift_section` so the headline banner can reference the model name.

### `_render_drift_section` (defects 1, 2, 3)
- Added a derived headline banner: when `latest_level == "red"` we emit `st.error("Performance degradation: drift threshold breached for {model} (drift status = RED, ...)")`; YELLOW emits `st.warning`. This replaces the old contradictory "No performance degradation detected" success banner.
- Wrapped Drift Alert Timeline, Mean PSI Over Time, and Mean KS Statistic Over Time in `drift_trend_guard(...)`. When the guard returns `(False, msg)` the chart is replaced with `st.info(msg)` plus either a single-row latest-snapshot dataframe (timeline) or a `st.metric` with the latest value (PSI/KS). No more 1.5 ms "trend" lines.
- Added `Last update: <ts>` `st.caption` under each drift chart, sourced from `_format_last_update(drift_history["timestamp"])`.

### `_format_last_update` (new helper)
- Pure formatter that coerces a series to `pd.to_datetime`, picks the max, and formats as `YYYY-MM-DD HH:MM:SS`. Returns `"unknown"` when empty/unparseable. Used by drift and throughput sections.

### `_render_performance_section` → Training Run History (defect 4)
- Computes `n_runs = len(mlflow_runs)` and runs `drift_trend_guard(mlflow_runs["timestamp"])`.
- When the guard fails (e.g. 3 runs over 0.1 ms), emits `st.info` and renders an **index-based bar chart** with x-axis `"Run 1" / "Run 2" / "Run 3"` and `barmode="group"`, `y_range=[0,1.05]` — never claims to be a temporal trend.
- When ≥5 runs and a real time span exist, the original timestamp-axis line chart is preserved.

### `_render_performance_alerts` (defect 1, secondary)
- Removed the green `st.success("No performance degradation detected for {model}.")` line — that was the literal banner the audit flagged.
- Replaced with an `st.caption(...)` that defers to the Drift Detection Overview banner as the authoritative health status. Cannot contradict a RED drift KPI.

### `_render_throughput_section` (defects 3, 6)
- Calls `_classify_throughput_freshness(...)` which inspects the latest timestamp in `scoring_throughput`. When the most recent sample is >24 h old, emits `st.warning("Historical (Oct 2024) — replace with live telemetry before production. Latest sample is N day(s) old.")` at the top of the section. This is the explicit "label as historical" path required by the audit.
- Adds `st.caption(f"Last update: {last_throughput_ts}")` under each of Scoring Throughput, Average Scoring Latency, and Scoring Error Rate.
- On Scoring Error Rate, adds an `add_hline` at `ERROR_RATE_SLO_TARGET` annotated `"SaaS SLO target (0.1%)"`.
- After the Throughput Summary KPI row, surfaces the SLO badge:
  - Breach: `st.error("SLO BREACH — error rate 1.03% is 10.3× the SaaS SLO target of 0.1%. Page on-call and open an incident.")`
  - Pass: `st.success("Error rate X% is within the 0.1% SaaS SLO target.")`

### `_classify_throughput_freshness` (new helper)
- Returns `(is_stale, message)` based on a 24-hour staleness threshold. Defensive: empty / unparseable timestamps return `(False, "")`. Handles tz-aware vs naive timestamps via `tz_localize(None)` in a try block.

### `_render_survival_section` (defect 5)
- **Removed entirely.** Survival analysis belongs to Page 07. The section header, the segment-color map, the figure builder, the median-survival summary table, and the dispatcher branches are all gone. No call site remains in `render_model_monitoring`.

---

## iter9 issues closed

| # | Defect | Fix location | Mechanism |
|---|---|---|---|
| 1 | Banner "No performance degradation detected" while Drift Status = RED | `_render_drift_section` + `_render_performance_alerts` | Banner now derived from `latest["alert_level"]` — RED triggers `st.error("Performance degradation: drift threshold breached…")`. Old success banner downgraded to caption that defers to drift status. |
| 2 | Drift "trend" charts on 1.5 ms x-axis (single point) | `_render_drift_section` | All three drift charts (Alert Timeline, Mean PSI, Mean KS) gated through `drift_trend_guard(...)`; sparse case renders `st.info(msg)` + snapshot row/metric instead of a misleading line. |
| 3 | Time-anchor split (May 2026 drift vs Oct 2024 throughput) | `_render_throughput_section` + `_render_drift_section` | Per-chart `Last update:` caption on every chart. `_classify_throughput_freshness` adds an explicit `Historical — replace with live telemetry before production` banner when the data is >24 h old. |
| 4 | Training Run History x-axis ~0.1 ms (3 calls plotted as series) | `_render_performance_section` | `drift_trend_guard` check on `mlflow_runs["timestamp"]`; sparse case switches to index-based bar chart (`Run 1 / Run 2 / Run 3`) and emits an `st.info` clarifying it is not a temporal series. |
| 5 | Page 07 KM curves duplicated on Page 08 (scope creep) | `render_model_monitoring` + `_render_survival_section` | Survival section call removed from main flow; `_render_survival_section` deleted; data-loader calls for survival data removed; page header simplified to "Model Monitoring". |
| 6 | Error rate 1.03% with no SLO breach indicator | `_render_throughput_section` | `ERROR_RATE_SLO_TARGET = 0.001`; Scoring Error Rate chart annotated with the SLO target line; post-KPI red `st.error` callout when `avg_error_rate > target` (with multiplier and on-call instruction); green confirmation when within target. |

---

## Constraints honored
- Only `src/dashboard/monitoring_view.py` was modified.
- F1 helper import is wrapped in `try/except` with a local fallback — page does not crash if `dashboard_helpers.drift_trend_guard` is absent.
- `render_model_monitoring(st_module, config, data_loader=None)` signature preserved.
- All other public/private function signatures preserved or extended only with optional parameters (`_render_drift_section` gained an optional `performance_alerts` arg with default `None`).
- Existing imports retained; new imports limited to `datetime` and the guarded helper.
- Python AST parse confirmed clean: `python -c "import ast; ast.parse(open(...).read())"` returned `OK`.

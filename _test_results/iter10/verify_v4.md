# Verification V4 — iter9 audit a4 (Pages 08 / 14 / 15)

Inputs verified:
- Issues source: `_test_results/iter9/iter9_audit_a4.md`
- Fix logs: `_test_results/iter10/fix_logs/{f1_helpers,f2_system_health,f3_monitoring,f5_app}.md`
- PNGs: `_test_results/dashboard_pages/{08_model_monitoring,14_mlflow_experiments,15_system_health}.png`
- Code: `src/dashboard/{monitoring_view.py,system_health_view.py,app.py,utils/dashboard_helpers.py}`

## Verdict table

| # | Issue (iter9 a4) | Owner | Verdict |
|---|---|---|---|
| 1 | P08 banner "No degradation" vs Drift RED contradiction | F3 | **FIXED** |
| 2 | P08 drift trend 1.5 ms x-axis (n=1) → `Insufficient history` | F1+F3 | **FIXED** |
| 3 | P08 throughput Oct 2024 vs drift May 2026 (19-mo gap) | F3 | **PARTIAL** |
| 4 | P08 Training Run History 0.1 ms x-axis | F3 | **FIXED** |
| 5 | P08 Kaplan–Meier curves duplicated from P07 | F3 | **FIXED** |
| 6 | P08 error rate 1.03% vs SLO <0.1% breach indicator | F3 | **FIXED** |
| 7 | P14 vs P15 MLflow status — single source of truth | F2+F5 | **FIXED** |
| 8 | P15 Experiments=0 vs Total Runs=3 same page | F2 | **FIXED** |
| 9 | P15 "All Systems Operational" vs Drift=RED propagation | F2 | **FIXED** |
| 10 | P14 3 runs LR=0.1 / training=1 s degenerate sweep | F5 | **NOT FIXED** |
| 11 | P14 Experiment Timeline 0.1 ms x-axis | F5 | **NOT FIXED** |

Score: 8 FIXED · 1 PARTIAL · 2 NOT FIXED · 0 REGRESSION (out of 11)

---

## Detailed findings

### #1 — P08 banner contradiction → FIXED
- **Code:** `src/dashboard/monitoring_view.py:142-178` derives the headline from `latest["alert_level"]`. RED ⇒ `st.error("Performance degradation: drift threshold breached for {model} (drift status = RED, ...)")`; YELLOW ⇒ `st.warning`. The old `st.success("No performance degradation detected for {model}.")` is downgraded to `st.caption` deferring to the drift overview (`_render_performance_alerts`, line 526–559).
- **PNG:** P08 top of page now shows a red "❌ Performance degradation: drift threshold breached for ensemble (drift status = RED ...)" banner, no green success line. Confirmed.

### #2 — P08 drift "trend" 1.5 ms x-axis → FIXED
- **Code:** `monitoring_view.py:178-321` wraps Drift Alert Timeline, Mean PSI Over Time, and Mean KS Statistic Over Time in `drift_trend_guard(drift_history.timestamp)`. When the guard returns `(False, msg)`, sparse-case rendering uses `st.info(msg)` plus a single-row snapshot dataframe / `st.metric`. Helper at `utils/dashboard_helpers.py` returns `Insufficient history — need ≥{N} observations, …`.
- **PNG:** P08 "Drift Alert Timeline / PSI Over Time / KS Over Time" now display a metric tile (`PSI=0.1521`, `KS=0.1129`) and a single-row snapshot rather than a 1.5 ms trend line. Confirmed — no degenerate ms-window plots remain on P08 drift section.

### #3 — P08 19-month time-anchor gap → PARTIAL
- **Code:** `monitoring_view.py:580-707` adds `Last update:` captions on Scoring Throughput / Avg Latency / Error Rate (lines 621/638/665) and `_classify_throughput_freshness(...)` triggers an `st.warning("Historical (Oct 2024) — replace with live telemetry before production. Latest sample is N day(s) old.")` at the top of the section when the latest timestamp is >24 h old.
- **PNG:** The throughput, latency, and error-rate charts ARE STILL RENDERED in the screenshot (green throughput area, orange latency, red error-rate spikes). The fix log says it adds a banner but the chart is still drawn (unlike P15 where stale data fully suppresses the chart per `system_health_view.py:794-811`).
- **Net:** Staleness is now LABELED on P08 (which closes the "no freshness disclosure" sub-defect), but the 24h Oct-2024 chart still sits next to the May-2026 drift snapshot — the 19-month visual contradiction is softened by captions, not eliminated. Hence PARTIAL.

### #4 — P08 Training Run History 0.1 ms x-axis → FIXED
- **Code:** `monitoring_view.py:467` runs `drift_trend_guard(mlflow_runs["timestamp"])`. When sparse, switches to **index-based bar chart** (x = `Run 1 / Run 2 / Run 3`, `barmode="group"`, `y_range=[0, 1.05]`) with an `st.info` clarifying it is not a temporal series.
- **PNG:** P08 "Training Run History" now shows an index-axis grouped bar chart (Run 1 / Run 2 / Run 3 across AUC/Precision/Recall/F1/Accuracy). No 0.1 ms timestamp axis. Confirmed.

### #5 — P08 KM curves duplicated from P07 → FIXED
- **Code:** `_render_survival_section` and the `survival_curves` / `survival_data` data-loader calls were removed from `monitoring_view.py`. Module docstring (`monitoring_view.py:67-68`) explicitly states *"Survival analysis (Kaplan-Meier curves) is intentionally NOT rendered here — it is owned by the dedicated Survival Analysis page (Page 07)."* Page header changed from "Model Monitoring & Survival Analysis" → "Model Monitoring".
- **PNG:** P08 contains no Kaplan–Meier section. Page ends with Throughput Summary + Monitoring Configuration. Confirmed.

### #6 — P08 error rate 1.03% vs <0.1% SLO → FIXED
- **Code:** `monitoring_view.py:24` defines `ERROR_RATE_SLO_TARGET = 0.001`. Scoring Error Rate chart adds `add_hline(y=0.001, annotation_text="SaaS SLO target (0.1%)")` (lines 650-657). After the KPI strip, `monitoring_view.py:694-707` emits `st.error("SLO BREACH — error rate 1.03% is 10.3× the SaaS SLO target of 0.1%. Page on-call and open an incident.")` when breached, otherwise `st.success`.
- **PNG:** P08 error rate chart shows a red dashed SLO line at the bottom; chart spikes clearly above. Throughput Summary KPI strip is followed by the SLO-breach treatment in code. Confirmed.

### #7 — P14 vs P15 MLflow single source of truth → FIXED
- **Code:** `app.py:119` defines module-level `_probe_mlflow_status(config)` shim that imports and delegates to `system_health_view.check_mlflow_health`. P14 banner at `app.py:4723-4748` consumes that result: connected ⇒ `st.success("Connected to MLflow tracking server")`; not-connected ⇒ `st.warning("MLflow tracking server not available — showing cached experiment data from artifacts. (...). Page 15 (System Health) will report the same status.")`. On the P15 side, `system_health_view.check_mlflow_health` was rewritten so connection requires `mlflow.search_experiments()` to succeed AND return ≥1 experiment — the old "sqlite-file-exists ⇒ HEALTHY" short-circuit is removed (`system_health_view.py:184` onward). New `mlflow_status_banner(...)` helper centralises copy.
- **PNG:** P14 banner: yellow/orange warning *"MLflow tracking server not available — showing cached experiment data from artifacts. ... Page 15 (System Health) will report the same status."*; P15 MLflow service card label: *"No (showing cached r…")* under "MLflow Tracking" header. Both pages now agree the server is down. Confirmed.

### #8 — P15 Experiments=0 vs Total Runs=3 → FIXED
- **Code:** `system_health_view.py:486-488` pre-loads `mlflow_runs_df = data_loader.load_mlflow_runs()` once, threads it into `_render_service_cards` (line 627) and `_render_mlflow_tracking` (line 805). MLflow card now reconciles `Experiments` via `max(len(live_experiments), distinct_experiments_from_runs, …)` with the invariant *"runs > 0 ⇒ experiments ≥ 1"*; both `Experiments` and `Total Runs` rendered together on the same card (lines 717-718).
- **PNG:** P15 "MLflow Tracking" card displays both `Experiments: 1` and `Total Runs: 3`. Same numbers also surface in the lower "Experiment Run History" block (Total Runs=3). No 0-vs-3 contradiction remains. Confirmed.

### #9 — P15 "All Systems Operational" vs Drift=RED → FIXED
- **Code:** `system_health_view.py:371-454` adds `_drift_alert_to_status` (red ⇒ `STATUS_DOWN`, yellow ⇒ `STATUS_DEGRADED`) and extends `get_system_health_summary` to query `data_loader.load_drift_history()`, fold a `drift` child into `services`, then propagate the worst-child state. Header copy at `system_health_view.py:583-585`: `STATUS_HEALTHY → "All Systems Operational"`, `STATUS_DEGRADED → "Degraded — Investigate Subsystems"`, `STATUS_DOWN → "System Issues Detected"`. Banner rendered with `st.warning` / `st.error` so icon, colour, and copy cannot disagree.
- **PNG:** P15 header is now red `❌ System Status: System Issues Detected` (NOT the green "All Systems Operational"). The drift card below shows `Current Drift Status: RED`. Aggregate now reflects the worst child. Confirmed.

### #10 — P14 degenerate sweep (3 runs, LR=0.1, training=1 s) → NOT FIXED
- **Code:** `app.py:4836-4867` "Hyperparameter Analysis" section (Learning Rate vs AUC, Epochs vs AUC) is unchanged from iter9 — three points clustered, no annotation. F5 fix log explicitly lists this as *not addressed* and only closes #15 (cross-page MLflow alignment). No "single config, sweep is illustrative" caption, no widening of the sweep (which is a model-layer change in `src/models/`).
- **PNG:** P14 "Hyperparameter Analysis" still shows three clustered points (LR≈0.1) with no caveat. AUC by Model still presents 0.0014 spread as a comparison. No annotation acknowledging degeneracy.

### #11 — P14 Experiment Timeline 0.1 ms x-axis → NOT FIXED
- **Code:** `app.py:4928-4953` "Experiment Timeline" still draws `px.scatter(runs_timeline, x="timestamp", y="auc", size="training_time_s", ...)` directly on a sub-millisecond timestamp axis. **No `drift_trend_guard` applied** here, no index-based fallback (contrast with the P08 fix at `monitoring_view.py:467` and the P13c fix at `app.py:4450/4578`). F5 closed `drift_trend_guard` on Page 13 tab c but missed the same pattern on Page 14.
- **PNG:** P14 "Experiment Timeline / Model Performance Over Time" still renders a scatter on a timestamp axis with three points; Plotly's auto-range visually widens the axis but the underlying single-instant cluster is unchanged.

---

## Cross-cutting observations

- **F1 helper `drift_trend_guard`** is correctly imported with try/except fallbacks in `monitoring_view.py:28`, `system_health_view.py:36`, `app.py:74` — none of the consumers crash when the helper is absent. The `Insufficient history — need ≥N observations` copy is uniform.
- **MLflow single source of truth (#7)** is implemented through a real shared probe (`_probe_mlflow_status` → `check_mlflow_health`), not just matching strings. This is a structural fix that survives future refactors.
- **`get_system_health_summary` worst-child propagation (#9)** is a structural rollup change (drift ⇒ DOWN forces `overall = STATUS_DOWN`), not just banner-text rewording.
- **The two NOT FIXED items (#10, #11)** are both on Page 14, both inside the MLflow Experiments view, and both are the same pattern that was successfully closed elsewhere — the P14 page just wasn't included in F5's drift-guard sweep. A localized follow-up (apply `drift_trend_guard` to the Experiment Timeline block + add a degenerate-sweep caption to Hyperparameter Analysis) would lift this to all-FIXED.
- **#3 PARTIAL** is the only fix where the screenshot diverges from the fix log claim: the fix log says "label as historical" via banner + caption, and that is true; but the chart is still drawn on a 24-h Oct-2024 axis next to a May-2026 drift snapshot — i.e., the same-page time-anchor split visually persists, just now annotated. F2's stricter approach for P15 (suppress the chart entirely when stale) was not mirrored on P08.

---

## 5-line summary

1. P08 (Model Monitoring) is fully cleaned up: banner now derives from drift status (RED ⇒ red error banner, no more "no degradation" contradiction), all three drift "trends" gated by `drift_trend_guard` with `Insufficient history` info + snapshot fallback, Training Run History switched to an index-based bar chart, Kaplan–Meier section deleted, and a `SLO BREACH — 1.03% is 10.3× target` red callout was added.
2. P15 (System Health) is fully cleaned up: header now reads `❌ System Issues Detected` because drift is folded into `get_system_health_summary` with worst-child propagation; same-page MLflow contradiction (Experiments=0 vs Total Runs=3) is reconciled by sharing one `mlflow_runs_df` and rendering both metrics on one card.
3. P14↔P15 MLflow status now passes through a single shared probe (`_probe_mlflow_status` → `check_mlflow_health`) — both pages display "tracking server not available / showing cached" in lockstep, with explicit cross-references.
4. The 19-month time-anchor split on P08 (#3) is **PARTIAL** — the fix labels the throughput chart as `Last update: <ts>` and adds a "Historical (Oct 2024)" warning banner, but the chart itself is still rendered (P15's stricter "suppress chart when stale" pattern was not applied on P08).
5. The two **NOT FIXED** issues are both on P14 (Hyperparameter Analysis still shows the degenerate LR=0.1 sweep with no annotation, and Experiment Timeline still scatter-plots three points on a 0.1 ms timestamp axis with no `drift_trend_guard`) — the same drift-guard pattern that closed #2 / #4 on P08 and the P13c block was not extended to P14.

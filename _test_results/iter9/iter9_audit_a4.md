# A4 — Monitoring / MLflow / System Health

Independent audit of pages 08, 14, 15. Sources: PNG screenshots + page_data MD dumps under `C:\Users\yoonc\Capstone-Design-1\_test_results\`.

---

## Page 08 — Model Monitoring

### Visible KPIs
- Drift: Total Checks=**1**, Current Status=**RED**, Red Alerts=1, Yellow Alerts=0
- Throughput: Avg Req/min=49.0, Peak=83.3, Avg Latency=19.1 ms, Avg Error Rate=**0.0103 (1.03%)**
- Best model by AUC: ensemble (AUC=0.8866)
- Headline banner: "No performance degradation detected for ensemble."

### Wrong / contradictory
- **Direct contradiction in the headline area:** banner says *"No performance degradation detected for ensemble"* while the very next KPI tile reads *Drift Status: **RED***. The page tells the user two opposite things in the same viewport.
- **"Trend" charts that span 1.5 milliseconds.** All three drift "Over Time" charts (Drift Alert Timeline, Mean PSI Over Time, Mean KS Statistic Over Time) have an x-axis from `11:56:50.282 → 11:56:50.2835` on May 10 2026 — a **~1.5 ms** window. With Total Checks=1, this is a single observation drawn as a trend line.
- **Training Run History x-axis is ~0.1 ms wide** (`11:56:50.2892 → 11:56:50.2893`). Three function calls plotted as a longitudinal series.
- **19-month time-anchor split on the same page.** Drift charts are stamped May 10 2026, while Throughput / Latency / Error Rate charts are stamped **Oct 15–16 2024**. Same monitoring page, two different epochs.
- **Error rate 1.03% is 10× the standard SaaS SLO target (<0.1%)** — yet the page surfaces no alert tied to it.

### Unreliable
- Drift "RED" derived from n=1 sample is statistically meaningless; PSI/KS thresholds (0.1/0.25, 0.05) cannot be evaluated on a single point.
- Throughput/latency/error charts on a 24-hour Oct 2024 axis cannot be the source for KPI tiles dated today; provenance is undefined.
- Page 07 Kaplan–Meier survival curves are duplicated here verbatim (scope creep — survival belongs to Survival Analysis, not Monitoring).

### Missing
- No drift severity over rolling windows; no per-feature PSI table.
- No alert rule audit trail (who/when acknowledged the RED).
- No rollback / canary / shadow-mode controls expected on a monitoring screen.
- No data freshness / lag indicator that would expose the 19-month gap to the operator.
- No SLO panel (target vs observed for latency, error rate, availability).

---

## Page 14 — MLflow Experiments

### Visible KPIs
- Banner: 🚨 *"MLflow tracking server **not available**. Showing cached experiment data from artifacts."*
- Total Runs=**3**, Best AUC=0.8866, Best Model=ensemble, Total Training Time=**3 s**
- Tracking URI: `sqlite:///mlflow/mlflow.db`, experiment=`churn_prediction`
- AUC by Model: ensemble 0.8866 / dl_model 0.8860 / ml_model 0.8852

### Wrong / unreliable
- **Page is in fallback mode** — the canonical MLflow page admits the tracking server is not reachable. Any "experiment history" on this screen is a static artifact dump, not a live registry.
- **3 runs is not an experiment history.** AUC range across all three runs = 0.8866 − 0.8852 = **0.0014** — degenerate spread; nothing learned about hyperparameters.
- **Hyperparameter "sweep" is fake.** Learning Rate vs AUC scatter plots three points clustered at LR=0.1 (per the run-detail table the runs share LR=0.1, training_time≈1 s) — there is no sweep, just one config repeated.
- **AUC per Training Second** chart: with all training times ≈1 s, the metric is just AUC again. Visual is uninformative by construction.
- **Experiment Timeline x-axis ≈ 0.1 ms wide** (`11:56:50.2892 → 11:56:50.2893`). Three sequential function calls dressed up as a temporal series.
- **Total Training Time = 3 s** for an end-to-end ML+DL+ensemble pipeline on n=20,000 is implausible — synthetic floor of 1 s per model.

### Missing
- No model registry stages (Staging / Production / Archived).
- No promotion / rollback / approval history.
- No run lineage metadata: training-data version, feature-set ID, code commit SHA, container image digest.
- No artifact size, no model signature, no input/output schema.
- No comparison against a previous champion; no challenger / shadow run.
- No tags, no run owner, no cost / GPU-hours.

---

## Page 15 — System Overview & Health

### Visible KPIs
- Header: "✅ System Status: **All Systems Operational**", 3/3 services healthy, last checked 2026-05-10T14:20:53.
- Redis Streaming: Connected=Yes, requests stream=**0**, responses stream=**0**.
- MLflow Tracking: Connected=**Yes**, Experiments=**0**, banner *"Connected to MLflow tracking server"* + *"No experiments found on MLflow server."*
- ML Pipeline: Artifacts=44, Models=4.
- Throughput: 49.0 req/min, 19.1 ms, 0.0103 error rate.
- Experiment Run History: Total Runs=**3**, Best AUC=0.8866, Total Train Time=3 s.
- Drift card: "Current Drift Status: **RED**".

### Wrong / contradictory
- **"All Systems Operational ✅" while Drift Status = RED** in a card immediately below. The aggregate header ignores the model-health subsystem.
- **Same-page contradiction on MLflow:** KPI tile says Experiments=**0**, but Experiment Run History on the same page says Total Runs=**3**. Both cannot be true.
- **All-green checkmarks on hollow services:** Redis "Connected: Yes" with `requests=0, responses=0`; MLflow "Connected: Yes" with `Experiments=0`. "Healthy" is reduced to "the TCP socket opened."
- **Time-anchor split repeated:** Scoring Throughput (24h) chart dated **Oct 15 2024**; Model Performance Over Time and PSI Drift Trend dated **May 10 2026** with ~0.1 ms / ~1.5 ms windows. Same page, two epochs.
- **Error rate 1.03% surfaced with no SLO breach indicator** — header still green.

### Unreliable
- Single global "Last checked" timestamp; no per-probe age, no probe latency, no consecutive-failure counter.
- "3/3 services healthy" is a count, not a health signal — does not include data freshness, queue lag, disk, memory, or model staleness.

### Missing
- No SLO/error-budget panel.
- No incident / alert log, no on-call info, no maintenance window banner.
- No infrastructure metrics (CPU, RAM, disk, network) for any host.
- No queue depth / consumer lag / DLQ size for Redis Streams.
- No model-staleness alarm (last train date vs threshold).
- No external dependency probe (DB, blob storage, identity provider).

---

## Cross-page contradictions (especially MLflow status across 14 and 15)

| # | Page A | Page B | Conflict |
|---|---|---|---|
| 1 | Page 14 banner: *"MLflow tracking server **not available**. Showing cached experiment data from artifacts."* | Page 15 banner: *"Connected to MLflow tracking server"* + KPI MLflow Connected=**Yes** | Direct opposite states of the same MLflow service on the same dashboard. |
| 2 | Page 15 KPI: MLflow Experiments=**0** | Page 15 Experiment Run History: Total Runs=**3** (and Page 14: Total Runs=3) | Same page (15) reports 0 and 3 simultaneously. |
| 3 | Page 08 banner: *"No performance degradation detected"* | Page 08 KPI + Page 15 card: Drift Status=**RED** | Headline contradicts the headline KPI. |
| 4 | Page 15 header: *"All Systems Operational ✅"* | Page 15 card: Drift Status=**RED**; Page 08 same | Aggregate health ignores model-health subsystem. |
| 5 | Page 08 / Page 15 throughput charts dated **Oct 15–16 2024** | Page 08 / Page 15 drift charts dated **May 10 2026** | ~19-month time-anchor split within the same dashboard view. |
| 6 | Page 14: 3 cached runs, server unavailable | Page 15: "3/3 services healthy", MLflow ✅ | Service health does not reflect the fallback admission on Page 14. |

---

## SLO checklist (target vs observed)

| Criterion | Target | Observed | Pass? |
|---|---|---|---|
| Error rate | <0.1% | **1.03%** (0.0103) | ❌ (10× over) |
| p50/avg latency | <50 ms | 19.1 ms | ✅ |
| Drift status | GREEN steady-state | **RED** (n=1 sample) | ❌ |
| Drift evidence base | rolling window (≥7 days, ≥hundreds of checks) | **Total Checks = 1** | ❌ |
| Monitoring data freshness | <5 min lag | Throughput dated Oct 2024 vs drift May 2026 — **~19 months gap** | ❌ |
| MLflow availability | live tracking server, ≥99.5% | Page 14 says "**not available**, cached" | ❌ |
| Experiment history depth | ≥10s of runs with real sweeps | **3 runs**, AUC spread 0.0014, all LR=0.1, 1 s training | ❌ |
| Model registry stages | Staging / Production / Archived visible | not present | ❌ |
| Service health probes | per-service age, latency, dependency probe | "Connected: Yes" + counter only | ❌ |
| Aggregate status truthfulness | green only when all subsystems green | Header "All Systems Operational" while Drift=RED | ❌ |
| Cross-page consistency (MLflow) | single source of truth | Page 14 says down, Page 15 says up | ❌ |
| SLO/error-budget panel | present on monitoring/health | not present | ❌ |
| Alert audit trail | ack/owner/runbook link on RED | not present | ❌ |
| Throughput | sustained ≥ design req/min | 49 avg / 83 peak (synthetic) | ⚠️ unverifiable on 2024-stamped chart |

Pass rate against an operations bar: **1 of 14**.

---

## SaaS-readiness verdict — Operations

**Verdict: NOT SaaS-ready. Hard fail on observability and lineage.** The monitoring surface contradicts itself within a single viewport (RED drift under a "no degradation" banner; "All Systems Operational" while Drift=RED), the MLflow story flips between pages 14 and 15, and the headline charts mix May-2026 millisecond windows with October-2024 24-hour windows. The error-rate KPI is 10× the conventional <0.1% SaaS SLO with no breach indicator. With Total Checks=1, three identical-config runs, training_time=1 s, and tracking server unreachable, what is shown is not telemetry — it is a one-shot synthetic snapshot styled as a live operations console. A buyer evaluating this for production would reject it on cross-page inconsistency alone.

**Top 3 blockers**
1. **MLflow status contradiction across pages 14 vs 15** (and within page 15: Experiments=0 vs Total Runs=3). One source of truth is non-negotiable for procurement; this kills model-lineage claims.
2. **Aggregate "All Systems Operational" while Drift Status = RED, plus the page 08 "no degradation" banner over a RED KPI.** Health rollups must propagate the worst child state, and headlines must not contradict their own KPIs.
3. **Time-anchor incoherence and degenerate trend windows** — drift "trends" on 1.5 ms x-axes, training timelines on 0.1 ms x-axes, and throughput charts 19 months out of phase with drift. Combined with the missing SLO/error-budget panel and the 1.03% (>10× target) error rate, the monitoring stack does not meet a baseline operations bar.

---

### 5-line summary
1. Page 08 contradicts itself: "No performance degradation detected" while Drift Status=RED, with all three "trend" charts spanning ~1.5 ms (single sample).
2. Page 08 also splits time anchors — drift charts at May 10 2026, throughput/latency/error charts at Oct 15–16 2024 — a ~19-month gap on one page.
3. Page 14 admits the MLflow tracking server is **not available** and shows 3 cached runs all at LR=0.1 / training=1 s — a degenerate, fallback "experiment history."
4. Page 15 directly contradicts Page 14 (MLflow Connected=Yes, "Connected to MLflow tracking server") and itself (Experiments=0 vs Total Runs=3), all under a green "All Systems Operational" header that ignores Drift=RED.
5. Operations SLO check: error rate 1.03% vs <0.1% target = 10× over; latency 19.1 ms passes; on a 14-criterion ops bar the dashboard scores 1/14 — **not SaaS-ready**.

# A4 — Model Monitoring / MLflow / System Health

> Source: independent visual audit of three PNGs only. No code consulted.

---

## Page 08 — Model Monitoring

### Visible KPIs
- Banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- Section header: "Model Monitoring & Survival Analysis"
- **Drift Detection Overview**
  - Total Checks: **1**
  - Current Status: **RED**
  - Red Alerts: **1**
  - Yellow Alerts: **0**
- **Drift Alert Timeline** chart
  - y-axis: "Number of Drifted Features", range 6 → 8
  - **Single red dot at y = 7**
  - x-axis ticks: `11:56:50.282`, `11:56:50.2825`, `11:56:50.283`, `11:56:50.2835` (sub-millisecond span on **May 10, 2026**)
- Section heading "PSI & KS Statistics Over Time" visible at fold (chart not yet rendered in screenshot crop).

### Wrong / suspicious
- **"Drift Alert Timeline" with 1 datapoint.** A timeline that contains exactly one observation is not a timeline; the chart shape implies history that does not exist.
- **x-axis spans ~1.5 milliseconds** (`.282` → `.2835` of the same second on the same day). This is the spacing of three consecutive function calls inside one process, not a time series.
- **"Total Checks: 1" but Current Status: RED.** A single check producing an alert means the simulator was deterministically configured to drift — not a real monitoring signal.
- **"Red Alerts: 1, Yellow Alerts: 0"** with Total Checks=1 implies 100% red-alert rate; an SRE buyer would read this as "the monitor only knows how to fire."

### Unreliable
- The page banner explicitly states **"All KPIs are simulator-generated"** — by the page's own admission, none of the drift counts represent production traffic.
- "Number of Drifted Features = 7" with no baseline window, no feature list, no PSI/KS thresholds visible above the fold.
- A "Survival Analysis" subtitle is in the section header but no survival curves, censoring counts, or hazard ratios are visible in the captured frame.

### Missing
- Drift window definition (rolling N hours? days?).
- PSI/KS threshold lines on the chart (typical: PSI > 0.2 = warn, > 0.25 = alarm).
- Per-feature drift table (which 7 features drifted?).
- Sample sizes for reference vs current population.
- Alert routing / acknowledgement state (who was paged, when, was it cleared).
- Historical retention (≥ 7 daily observations needed to call something a "trend").

---

## Page 14 — MLflow Experiments

### Visible KPIs
- Banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- **MLflow Configuration JSON**
  - `tracking_uri`: `"sqlite:///mlflow/mlflow.db"`
  - `experiment_name`: `"churn_prediction"`
  - `log_models`: `true`
  - `log_artifacts`: `true`
- **Status banner (info-blue):** "MLflow tracking server not available. Showing cached experiment data from artifacts."
- **Experiment Run History**
  - Total Runs: **3**
  - Best AUC: **0.8866**
  - Best Model: **ensemble**
  - Total Training Time: **3s**
- **Run table (3 rows):**
  | # | timestamp | model_type | auc | auc_roc | accuracy | precision | recall | f1_score | f1 | run_id | training_time_s | params_lr |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 2026-05-10T11:56:50.289248+00:00 | ml_model | 0.8852 | 0.885247 | 0.8209 | 0.5331 | **0.7791** | 0.6331 | 0.633067 | current_0 | 1.0 | 0.1000 |
  | 1 | 2026-05-10T11:56:50.289284+00:00 | dl_model | 0.8860 | 0.885973 | 0.8671 | **0.6759** | 0.6318 | **0.6531** | 0.653093 | current_1 | 1.0 | 0.1000 |
  | 2 | 2026-05-10T11:56:50.289288+00:00 | ensemble | **0.8866** | 0.886606 | 0.8602 | 0.6426 | 0.6621 | 0.6522 | 0.652239 | current_2 | 1.0 | 0.1000 |

### Wrong / suspicious
- **"MLflow tracking server not available"** displayed in a calm info-blue banner, not as an error. For a paid SaaS observability surface this is a P1 broken integration, not an informational notice.
- **All 3 runs share `params_lr = 0.1000`** and all share `training_time_s = 1.0`. Plotting these as a "Learning Rate vs AUC" comparison yields three points stacked on x = 0.1 — a degenerate sweep (no actual sweep occurred).
- **All 3 timestamps are within 40 microseconds** (`...50.289248`, `...50.289284`, `...50.289288`). These are not three training runs; they are three rows written by the same Python process in the same millisecond. "Total Training Time: 3s" is therefore the sum of three ~1s claims, not a real wall-clock training history.
- **`auc` and `auc_roc` are nearly identical but not equal** (e.g. row 0: 0.8852 vs 0.885247). Either it is the same metric reported twice (then they should match) or two different metrics (then the labels are misleading).
- **Best AUC differential is 0.0014** across "ml", "dl", and "ensemble" — a meaningless gap that would not survive a single resample. Promoting "ensemble" as Best Model on Δ=0.0006 vs dl_model is statistical theatre.
- **`tracking_uri = sqlite:///mlflow/mlflow.db`** — a local SQLite file is not an MLflow tracking server in the SaaS sense; it is a single-process file lock. Multi-tenant observability is impossible.

### Unreliable
- "Showing cached experiment data from artifacts" means the surface is not live; what the buyer sees may be hours or days stale with no freshness indicator.
- `run_id` values are `current_0`, `current_1`, `current_2` — synthetic ordinals, not MLflow UUIDs. Real MLflow IDs are 32-hex strings.
- `precision` for ml_model = **0.5331** with recall **0.7791** ⇒ heavily skewed; promoting any of these models is premature regardless.

### Missing
- MLflow server uptime / last-heartbeat timestamp.
- Experiment lineage (parent run, dataset version, code commit hash).
- Hyperparameter ranges actually swept (only `params_lr` partly visible, all = 0.1).
- Confusion matrices, calibration curves, threshold sweeps.
- Cross-validation fold metrics or std-dev across seeds.
- Model registry stage info (Staging / Production / Archived).
- Reproducibility metadata: git commit, dataset hash, environment hash.

---

## Page 15 — System Health

### Visible KPIs
- Banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- Title: "System Overview & Health"
- Subtitle: "Real-time health monitoring for all system components: streaming pipeline, ML tracking, and model serving."
- **System Status: All Systems Operational**
- Last checked: **2026-05-10T13:45:27.047556**
- **3/3 services healthy** (full blue progress bar)
- **Service Health cards (all green check):**
  - **Redis Streaming** — Connected: **Yes**, Stream (requests): **0**, Stream (responses): **0**
  - **MLflow Tracking** — Connected: **Yes**, Experiments: **0**
  - **ML Pipeline** — Artifacts: **44**, Models: **4**

### Wrong / suspicious
- **"MLflow Tracking — Connected: Yes"** on this page directly contradicts page 14's banner **"MLflow tracking server not available."** Two surfaces of the same dashboard disagree about the same dependency at the same time. This alone is a credibility-killer.
- **Redis Streaming: requests = 0, responses = 0** but the card is green/healthy. A real-time scoring stream that has processed zero messages should not be reported as "healthy" — it is either idle (yellow / "no traffic") or disconnected (red).
- **MLflow Tracking: Experiments = 0** — but page 14 shows 3 runs under experiment "churn_prediction". Either this counter is wrong, or page 14 is reading from a different store (consistent with the "cached from artifacts" message).
- **"All Systems Operational"** is a hard-coded boolean: every signal under it is either zero or contradictory; no probe, no latency, no error budget is plotted.
- **Last-checked timestamp is at sub-microsecond precision** (`...27.047556`). Health checks are seconds-scale; surfacing 6 fractional digits is decorative.

### Unreliable
- The whole page is at-a-glance green while two of three "healthy" services have empty traffic counters and one is contradicted by another page. Health bars without underlying probes are pure decoration.
- No DB connection ping, no model-file freshness, no last-successful-prediction timestamp, no queue lag — the buyer-grade signals listed in the rubric are entirely absent.
- "Artifacts: 44 / Models: 4" is shown as an indicator of pipeline health, but those are inventory counts, not health metrics.

### Missing
- Per-service: uptime %, p50/p95/p99 latency, error rate, success-rate over rolling window.
- Probe age / staleness ("data more than 5 min old").
- Disk, memory, CPU saturation for the host.
- Last successful training run, last successful prediction, last drift check.
- Alert/incident history (was anything fired in the last 24h?).
- DB / Redis connection pool stats; queue depths; consumer-lag.
- Version / build hash of each service so on-call knows what is deployed.

---

## SaaS-readiness verdict — Operations

**Verdict: DO-NOT-SHIP** (paid tier).
A second, weaker option (NEEDS-DISCLAIMER) only applies if every page is rebadged "Demo / Synthetic" and the MLflow contradiction between Page 14 and Page 15 is resolved before any external eyes see it.

### SLO checklist (target vs observed)
| Signal | SaaS target | Observed | Pass? |
|---|---|---|---|
| Error rate | < 1% (2-nines), < 0.1% (4-nines) | **Not surfaced** anywhere | FAIL |
| Latency p50 / p95 | < 50ms / < 100ms | **Not surfaced** | FAIL |
| Throughput | > 0 requests in last window | Stream (requests) = **0**, (responses) = **0** | FAIL |
| Drift window | ≥ 7 observations | **1** observation, ~1.5 ms span | FAIL |
| MLflow uptime | live tracking server, ≥ 99.5% | "tracking server not available — cached"; SQLite file backend | FAIL |
| Health-check probes | per-dependency probe + age | hard-coded green badges, contradicted across pages | FAIL |

### Top 3 operations blockers
1. **MLflow is offline and the dashboard disagrees with itself.** Page 14 says "tracking server not available, showing cached." Page 15 says "MLflow Tracking — Connected: Yes." A buyer who clicks both pages in the same session loses trust in every other number on the site. Stand up a real tracking server (not `sqlite:///`), surface a live heartbeat, and unify the health probe.
2. **The "Drift Alert Timeline" is a single dot whose x-axis spans 1.5 milliseconds.** This is not a trend, not a window, and not actionable. Refuse to render a timeline below 7 observations and show an "Insufficient history (N=1)" placeholder. Define and display the drift window (rolling 24h / 7d), PSI and KS thresholds, and the per-feature breakdown.
3. **System Health is decoration, not telemetry.** All three service cards are green while two of them report zero traffic and one is contradicted on a sibling page. Replace the static checkmarks with actual probes — DB ping latency, Redis XLEN / consumer lag, model-file mtime, last successful prediction timestamp, error rate over rolling window — and let the badge color be derived from those numbers, not hard-coded.

Secondary, but still blocking for paid: the MLflow "hyperparameter sweep" of three runs all at `params_lr=0.1` plotted as Learning-Rate-vs-AUC is a degenerate scatter (all points on x=0.1, ΔAUC = 0.0014). Either run an actual sweep or remove the comparison chart.

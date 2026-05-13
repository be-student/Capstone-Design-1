## G6 — Infra / Realtime / MLflow

### Test results
| Suite | Tests | Pass | Fail | Skip | Duration | Status |
|---|---:|---:|---:|---:|---:|:--|
| G6 | 730 | 456 | 2 | 0 | 10.59s | RED (272 errors + 2 fails) |

Note: pytest reports `2 failed, 456 passed, 7 warnings, 272 errors` — the 272 errors are setup-time (collection-stage) errors for parametrized fixtures, not runtime failures. Effective collected suite is 730 tests; only 458 reached actual execution.

**Failure breakdown:**
- MLflow `Run already active` leak: **0** — heads-up note mentions a known 42-failure leak, but this iteration shows ZERO instances of that signature. The MLflow leak appears to have been masked by an earlier-stage fixture error (see below) that prevents any MLflow run from starting in the first place.
- `test_clv_predictions_positive` LARGE-bound: **0** — not in scope of group 6 files (CLV-prediction tests are in another group).
- New / real failures: **2 + 272 fixture-error cascade**, all rooted in **two Windows-platform issues**:
  1. **`UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2`** — affects 272 setup errors plus the `test_dockerfile_syntax_no_empty_from` failure. Root cause: `open(path)` / `Path.read_text()` calls in fixtures (e.g. `tests/test_mlflow_tracking.py:124`, `tests/test_redis_consumer.py:41`, `tests/test_scoring_api.py:44`, `tests/test_docker_setup.py:685`) load `config/simulator_config.yaml` and `Dockerfile.dashboard` without `encoding="utf-8"`, so Python 3.14 on Windows uses the locale default (cp949 on this Korean-locale machine). The YAML/Dockerfile contain UTF-8 multi-byte chars (`0xe2` — likely an em-dash or arrow), causing every parametrized fixture in 5 suites (`test_mlflow_tracking`, `test_model_registry`, `test_redis_consumer`, `test_redis_streaming`, `test_scoring_api`) to fail at setup. **Reproducible, deterministic, Windows-only**.
  2. **`tests/test_entrypoint.py::test_script_syntax_valid`** — `bash -n` fails to find `scripts/entrypoint.sh`; stderr shows the path mangled as `C:UsersyooncCapstone-Design-1scriptsentrypoint.sh` (backslashes stripped). The test passes a Windows-native path to Git-Bash without conversion. **Real Windows-only test bug**, not a product defect.

These are NOT product regressions. They are Windows-specific test-infrastructure issues that did not surface in the Docker-based CI runs (Linux locale = UTF-8). They DO mean the Windows developer-experience for running infra/MLflow tests locally is broken.

### Dashboard slice

**Page 08 — Model Monitoring & Survival Analysis**
- KPIs: Total Checks 1, Status GREEN, Red 0, Yellow 0, Avg req/min 49.0, Peak 83.3, Avg latency 19.1 ms, Avg error rate 0.0103.
- Drift Alert Timeline + Mean PSI + Mean KS — every chart has exactly ONE data point (timestamp 2026-05-08T10:33:20). Trend visualisations are meaningless with n=1; the page should render an "Insufficient history (need ≥7 observations)" message instead.
- "Model Metrics Across Training Runs" plots 3 timestamps that are within ~35 microseconds of each other on a time axis — three sequential function calls dressed up as longitudinal data.
- Throughput / latency / error charts each have 48 half-hour points with realistic-looking ranges. The shape is plausible but the underlying source is the synthetic generator (banner confirms SMALL mode).
- Page title bundles "Model Monitoring & Survival Analysis" — the KM survival curves are duplicated from page 07 (scope creep).

**Page 13 — Real-Time Scoring (3 tabs consolidated)**
- Service-Health KPIs: Redis Connected, Request Stream **0**, Response Stream **0**, Consumer Group `scoring_consumers`.
- Live-Scoring KPIs: Total Scores 200, Avg Churn Prob 27.30%, High/Critical Risk 17, Primary Model ensemble.
- Offers tab: 44 offers, Total Cost 1,196,659 KRW, Expected Revenue Saved 10,752,341 KRW, ROI 8.0x.
- Monitoring tab: Total Drift Checks 1, Latest GREEN, Red 0, Yellow 0.
- Inconsistency: streams report depth 0 but Total Scores = 200 — labels conflate "queue depth right now" with "lifetime processed", which will confuse operators.
- "Scoring Volume Over Time" is **uniformly 4** across all 50 hourly buckets — a flat line that no real production traffic ever produces. Confirms the data is generator output, not Redis-derived.
- Drift charts again single-point (same as page 08).

**Page 14 — MLflow Experiments**
- KPIs: Total Runs 3, Best AUC 0.8826, Best Model ensemble, Total Training Time 3s.
- **Banner: "MLflow tracking server not available. Showing cached experiment data from artifacts."** — page is permanently in fallback mode in this build. Cache-fallback works (the page renders), but no test in `test_mlflow_tracking.py` actually exercises the dashboard fallback path; the suite tests the tracker library, not the dashboard's degraded-mode UX.
- All 3 runs share **learning_rate=0.1, epochs=1**. Hyperparameter scatter ("Learning Rate vs AUC", "Epochs vs AUC") collapses to a single x-coordinate with 3 stacked points — useless as a sweep visualisation.
- "MLflow Run Performance Comparison" radar reports n=0 for every trace — chart is broken (renders empty radar).
- "Model Performance Over Time" — same 3 microsecond-spaced timestamps as page 08.
- 3 runs total = no real experiment history; this is a one-shot training pass dressed as an experiment-tracking surface.

### SaaS production-readiness checklist
| Criterion | Target | Observed | Pass? |
|---|---|---|:-:|
| Avg latency | <50 ms | 19.1 ms (synthetic) | PASS (number, not source) |
| P95 / peak latency | <100 ms | ~23 ms peak (synthetic) | PASS (number, not source) |
| Error rate | <0.1% | 1.03–1.86% | FAIL |
| Throughput | scalable, real | 49 req/min avg, "Volume = 4" flat synthetic | FAIL |
| MLflow HA | always-on tracking server | offline, dashboard in cache-fallback mode | FAIL |
| Drift window | ≥7 days history | 1 point (single timestamp) | FAIL |
| Drift thresholds enforced | tested PSI<0.1, KS<0.05 | observed PSI 0.0064 / KS 0.0196; threshold logic IS tested in `test_drift_detection.py` and `test_ks_drift.py` (those subsets pass) | PASS (logic only) |
| Hyperparameter sweep | real grid/Bayesian search | 3 runs all at LR=0.1 / epochs=1 | FAIL |
| Redis streams live | producer + consumer measurable | streams empty (depth 0); cannot tell from dashboard whether the consumer is actually attached or stubbed | FAIL (unverifiable) |
| Test suite green on dev OS | green on Win + Linux | Windows: 272 setup errors + 2 fails from cp949 / path bugs; Linux/Docker: presumed green | FAIL (Windows DX) |
| Synthetic-data disclosure | banner present | banner present on every page | PASS |

### SaaS-readiness verdict — Operations domain

**Verdict: DO-NOT-SHIP**

**Rationale:**
The operations surface is honest about being a demo (synthetic-data banner, MLflow fallback notice) but the gap between what an operator NEEDS to run a SaaS and what is wired up is too large to ship even with a disclaimer. The MLflow tracking server is not running in the captured build, so real experiment history, model lineage, and stage transitions are not auditable from the dashboard — only a 3-run cached snapshot is shown. Every drift chart has one data point. The "Scoring Volume Over Time" series is a constant 4, which is not a credible representation of real traffic and immediately telegraphs to a buyer that no production telemetry pipeline is connected. Error rate at ~1.5% is 15× the typical SaaS SLO. Two test files cannot even be collected on the developer's own Windows machine, so the local TDD loop is broken for anyone working on infra. None of these are individually unfixable, but together they describe a system that is one or two iterations short of a credible operations story.

**Top deployment blockers (in priority order):**
1. **Stand up a real MLflow tracking server** (Docker compose service is present in code per backend-config tests, but the captured dashboard run could not reach it). Without a live server there is no model registry, no stage transitions, no audit trail.
2. **Fix the single-point drift charts** — either accumulate ≥7 days of monitoring observations before rendering trend charts, or replace the chart with an "Insufficient history" placeholder. Shipping a "trend" with n=1 is misleading.
3. **Replace the synthetic "Scoring Volume = 4" series with real Redis-stream-derived metrics**, and reconcile the Request/Response Stream depth KPIs (currently 0 alongside Total Scores = 200, which is internally inconsistent and undermines operator trust).
4. **Drive observed error rate from ~1.5% to <0.1%** before any SaaS pricing conversation. The dashboard normalises a number that would burn through any error budget in hours.
5. **Run a real hyperparameter sweep**: 3 MLflow runs all at LR=0.1 / epochs=1 is not an experiment history, it is a smoke test. Either remove the "Hyperparameter Analysis" section or populate it with an actual grid/Bayesian sweep.
6. **Fix Windows test-collection cp949 bug** (`encoding="utf-8"` on every YAML / Dockerfile read in tests) and the `entrypoint.sh` path conversion in `test_entrypoint.py`. 274 broken Windows tests is a developer-experience blocker, not a product blocker, but it slows every fix above.

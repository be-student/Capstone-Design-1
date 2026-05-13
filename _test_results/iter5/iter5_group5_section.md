## G5 — Dashboard Views / UX

### Test results
| Suite | Tests | Pass | Fail | Skip | Errors | Status |
|---|---:|---:|---:|---:|---:|---|
| test_churn_analytics_views.py | 24 | 24 | 0 | 0 | 0 | PASS |
| test_churn_uplift_segmentation_views.py | 33 | 33 | 0 | 0 | 0 | PASS |
| test_clv_cohort_views.py | 39 | 39 | 0 | 0 | 0 | PASS |
| test_dashboard.py | 35 | 32 | 3 | 0 | 0 | FAIL (env) |
| test_dashboard_helpers.py | 20 | 18 | 0 | 0 | 2 | ERROR (env) |
| test_model_monitoring_view.py | 27 | 5 | 0 | 0 | 22 | ERROR (env) |
| test_streamlit_dashboard.py | 70 | 11 | 0 | 0 | 59 | ERROR (env) |
| test_survival_recommendations_views.py | 46 | 19 | 0 | 0 | 27 | ERROR (env) |
| test_system_health_view.py | 27 | 0 | 0 | 0 | 27 | ERROR (env) |
| **Aggregate (junit)** | **652** | **378** | **3** | **0** | **271** | 19.06s |

**Failure root cause (single, identical for all 274 failed/errored tests):**
`UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 2057` originating in `src/dashboard/app.py:83`:
```python
with open(CONFIG_PATH, "r") as f:        # <-- no encoding='utf-8'
    return yaml.safe_load(f)
```
The YAML config contains UTF-8 multibyte content (em-dashes / Korean / unicode arrows). On a Windows host whose system default codepage is `cp949` (Korean — this auditor's machine), the file opens with the locale codec and crashes. Tests in production-style Docker (Linux, UTF-8) pass; tests on Korean/Japanese/Chinese Windows dev boxes catastrophically fail. This is **a real product defect for any non-Linux SaaS dev/CI environment**, but it is masked when the smoke gate is Docker-only.

Pure dashboard-data-and-render tests (the 3 view-test suites that fully passed: `test_churn_analytics_views`, `test_churn_uplift_segmentation_views`, `test_clv_cohort_views`) never touched the YAML loader, hence 96 view-render assertions all green.

### iter2_overview audit
Visual diff of the three root-level PNGs:

| PNG | Synthetic-data banner present? | Title visible? | Notes |
|---|---|---|---|
| `iter2_overview_banner.png` (initial fresh deploy) | **NO** | yes | banner missing — clean Overview layout, jumps straight from page chrome to "Churn Prediction Overview" H1. |
| `iter2_overview_after_refresh.png` (browser reload) | **YES** | yes | yellow warning bar present: "Synthetic data — SMALL mode (n=5000). Numbers shown are illustrative; they do NOT represent production performance. Group-size validation: FAILED." rendered above H1. |
| `iter2_overview_after_restart.png` (container restart) | **NO** | yes | banner gone again, identical layout to the fresh-deploy capture. |

**Hot-reload fix verification (iter3 Dockerfile patch — `runOnSave=true`, `fileWatcherType=poll`):** does NOT resolve the issue documented here. The iter3 patch fixes Streamlit's source-file hot reload for code edits — it has no effect on this banner being a stateless render-time decision driven by a config flag that gets read once at app boot. The PNG sequence in iter2 is the authoritative evidence: banner is intermittent, present only on the *post-refresh* state, gone on both *fresh* and *restart*. That points to `st.session_state` or the cached `load_config()` momentarily showing the warning during a re-render, then the Streamlit page-render loop suppressing it on full re-runs. **Not fixed.**

### Cross-page UX audit (SaaS exec-demo perspective)

| Area | Status | Detail |
|---|---|---|
| Synthetic-data banner persistence | **FAIL** | Inconsistent across refresh/restart (see PNG audit above). For an external SaaS demo this is the single most important disclosure and it disappears 2 of 3 times. |
| Group-size-validation = FAILED text | **OK as-is** | Correct disclosure for SMALL mode, not a bug. Phrasing could be softer ("synthetic-only validation skipped") for non-technical viewers. |
| KPI definition consistency | **FAIL** | "Overall ROI" reported as 5.2× (page 05), 9.0× (page 09), 3.4× (page 12). No single source of truth. None of the dashboard tests in scope assert ROI cross-page equality. |
| Segment taxonomy consistency | **FAIL** | Behavioral segments (vip_loyal/regular_loyal/dormant/explorer/new_customer/bargain_hunter, 6 names) coexist with uplift segments (high_value_sure_thing/high_value_persuadable/sleeping_dog/lost_cause, 4 names). They appear interchangeably across pages 03/05/09/11/12 with no legend reconciling the two taxonomies. |
| Empty-state handling | **FAIL** | Page 06 A/B Testing: 1 chart, all-zero KPIs, no "No experiments logged yet" empty state. Page 12 Campaign Effectiveness radar: n=0 traces — chart frame renders blank instead of empty-state. Page 14 MLflow Run radar: n=0 traces — same. Page 10 CLV-vs-Churn scatter: x_sample=null for first ~5 of every trace. |
| Format / precision leak | **FAIL** | Page 12: "Customers Retained = 125.66457549932906" (raw float, 14 decimals — should be `126` integer or `125.7`). |
| Misleading time-axis charts | **FAIL** | Page 08 drift trend: 1 datapoint shown as a "trend over time". Page 13: 1 datapoint trend. Page 14 "Model Performance Over Time": 3 timestamps within microseconds of each other (3 sequential function calls, not a real time series) plotted as a temporal line chart. |
| Page 13 internal contradiction | **FAIL** | "Request Stream 0" KPI card is rendered next to "Total Scores 200" KPI card. Either the stream depth metric is broken or the scores-served counter is reading from a stale cache; either way the two cards visibly contradict each other. |
| Hot-reload fix (iter3 Dockerfile) | **NOT VERIFIED** | The patch addresses code-edit hot-reload only; the iter2 PNGs show the banner is still intermittent across browser-refresh + container-restart. |
| Tests enforcing single source of truth | **FAIL** | None of the 9 scoped suites assert: (a) banner appears on every page, (b) ROI matches across pages 05/09/12, (c) segment taxonomy is consistent across pages, (d) empty states render when n=0. The view-render tests confirm that *charts render* but not that *cross-page KPIs reconcile*. |

### SaaS-readiness verdict — Dashboard UX

**Verdict:** **DO-NOT-SHIP** for an external SaaS pilot. SHIP only as an internal capstone demo with an explicit verbal disclaimer.

**Rationale:**
1. The synthetic-data banner — the *single piece of disclosure that makes this dashboard ethically defensible* given the FAILED group-size validation — is intermittent across refresh and restart. iter2 PNG evidence directly contradicts the iter3_final.md claim that the hot-reload patch fixed the banner; the patch addresses a different problem.
2. Three different ROI numbers on three pages of the same product, with no reconciling legend, is a hard credibility blocker for any executive demo and a hard liability blocker for any SaaS pilot where customers will quote the dashboard back to their finance team.
3. Production smells visible to any prospect on a 5-minute walkthrough: a 14-decimal float on the campaign page, single-point "trend over time" charts, "Request Stream 0" next to "Total Scores 200", and broken radar charts on pages 12 and 14.
4. The product-side `load_config()` call at `src/dashboard/app.py:83` opens YAML without `encoding='utf-8'` — this is a real defect for SaaS customers running on Windows hosts with non-UTF-8 system codepages (Korean, Japanese, Chinese), and it caused 274 of 652 tests to fail in this audit. Docker masks it; a `pip install + streamlit run` install path on Windows does not.
5. View-rendering test coverage is healthy (96 of the pure-render assertions pass) but **zero** tests enforce cross-page invariants (banner present on every page, single ROI definition, consistent segment taxonomy, n=0 empty states). The test suite is locally green but globally blind.

**Top UX blockers for an external SaaS pilot:**
1. **Banner is not persistent.** Add a render-time guard that re-checks the synthetic-data flag on every page-render, not just at app boot. Cover with a Playwright test that asserts the banner element is present after `page.reload()` and after a container restart. (This must include the synthetic-data flag wiring, not just the Streamlit hot-reload fix.)
2. **Single source of truth for ROI.** One `compute_overall_roi()` helper called by pages 05, 09, 12; assert equality in a cross-page test.
3. **Empty-state library.** Pages 06, 10, 12, 14 currently render broken/blank charts when `n=0`; replace with a shared `render_empty_state(message, suggested_action)` component.
4. **Number-formatting helper.** A `format_count(x)` that floors and adds thousand separators; ban raw `f"{x}"` for KPI cards in lint/CI.
5. **Drift "trend" guard.** When data has fewer than ~5 timestamps, render a `st.info("Insufficient history for trend analysis")` instead of a 1-point line chart.
6. **`encoding='utf-8'` everywhere.** Audit every `open(...)` call in `src/dashboard/` for missing encoding kwarg; fail CI if any are unspecified. This unblocks Windows dev environments.
7. **Cross-page integration tests.** Add a test that loads each of the 14 pages, asserts the synthetic-data banner element is present, and asserts that any KPI labelled "Overall ROI" equals the same value across pages. None of the 9 scoped suites currently do this.

## G3 — A/B / Cohort / Uplift

### Test results
| Suite | Tests | Pass | Fail | Skip | Duration | Status |
|---|---:|---:|---:|---:|---:|---|
| test_ab_statistical_methods.py | 46 | 27 | 0 (19 errors) | 0 | 3.00s | RED (env) |
| test_ab_testing.py | 42 | 0 | 0 (42 errors) | 0 | 0.10s | RED (env) |
| test_cohort_analysis.py | 87 | 87 | 0 | 0 | 4.28s | GREEN |
| test_cohort_computations.py | 49 | 47 | 2 | 0 | 3.57s | RED (env) |
| test_experiment_manager.py | 46 | 1 | 0 (45 errors) | 0 | 3.04s | RED (env) |
| test_statistical_testing.py | 54 | 54 | 0 | 0 | 2.91s | GREEN |
| test_uplift.py | 22 | 2 | 0 (20 errors) | 0 | 2.86s | RED (env) |
| test_uplift_model.py | 35 | 0 | 0 (35 errors) | 0 | 0.08s | RED (env) |
| **TOTALS (Group 3)** | **381** | **218** | **2** + **161 errors** | **0** | **5.81s** | **RED** |

JUnit XML written to `_test_results/iter5_group3.xml`.

**Root cause of all 161 errors and both 2 failures:** `UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 2057`. Every fixture (and the two `TestCohortDashboardIntegration` tests) opens `config/simulator_config.yaml` via plain `open(CONFIG_PATH, "r")` without `encoding="utf-8"`. On Korean Windows the default codec is cp949, which fails on the UTF-8 byte sequence (likely an em-dash or arrow). The product code is fine; this is a test-infrastructure portability bug. Affected fixtures: `tests/test_uplift.py:34`, `tests/test_uplift_model.py` (parallel), `tests/test_ab_testing.py` (parallel), `tests/test_experiment_manager.py` (parallel), `tests/test_cohort_computations.py:613` (and a sibling). Logic-bearing tests that DON'T touch the YAML config (the 218 that pass) all green — including the entire `test_cohort_analysis.py` (87/87) and `test_statistical_testing.py` (54/54) suites.

### Dashboard slice

**Page 04 (Cohort Analysis)** — KPIs: Total Cohorts **2**, Periods Tracked **6**, Avg Period-1 Retention **98.7%**, Avg Final Retention **41.4%**. Heatmap shows only 2024-01 / 2024-02; retention curves both end ≈ 90% by period 4. Observations:
- Only **2 cohorts** is far below SaaS norm (6-12 monthly cohorts is typical for a YoY view). With 2 cohorts the heatmap, period-over-period delta bar, and "Retention Curves by Cohort" all degenerate into pairs of lines — the chart is structurally underpowered for any longitudinal insight.
- **KPI vs chart mismatch**: KPI claims `Avg Final Retention 41.4%` but the rightmost values on both retention curves and the average curve are ≈ 90%. The 41.4% likely represents a longer-horizon model output that isn't plotted. No test in Group 3 asserts that the displayed chart's terminal value matches the KPI card.

**Page 06 (A/B Testing Results)** — KPIs: Total Experiments **0**, Significant Results **0**, Best Experiment **N/A**, Avg Lift **0.0%**, Required Sample Size (per group) **906**, Total Participants Needed **1,812**, Expected Duration **19 days**. Single chart on the entire page (Power vs Sample Size). Observations:
- **Empty-state regression**: zero experiments produces a wall of zero-valued KPI cards. To a non-technical SaaS buyer this reads as "the model is broken" rather than "no experiments yet". A SaaS-grade UX would render an explicit empty-state ("Run an experiment to populate this view") and hide the zero KPIs.
- **Lift disclosure absent**: the page surfaces no notice that historic synthetic runs in this codebase reported **33.8% observed lift** (per `_test_results/iter3_final.md`), an order of magnitude above the 5–15% literature norm. Nothing on the page would prevent a buyer from anchoring on that figure.
- The previously-noted "Group-size validation: FAILED" banner is the only data-quality signal.

**Page 11 (Uplift Modeling Results)** — KPIs: Avg Uplift Score **0.0714**, Avg Treatment Effect **0.0714**, Persuadable Customers **4,143**, Sleeping Dogs **857**. Six charts. Observations:
- **Uplift collapse FIXED** — distribution histograms show real spread (orange uplift histogram and purple treatment-effect histogram both display variation; no longer a single spike). Verified at the test level by `test_uplift_scores_have_variance` (`np.std > 0.01`) and `test_uplift_both_positive_and_negative`. Note: there is **no test asserting `>= 1000` unique values**; the variance gate is the only protection against re-collapse.
- **Uplift == Treatment-Effect duplication BUG STILL PRESENT**: the two top KPIs match to 4 decimals (`0.0714 == 0.0714`); the "Uplift Score vs Treatment Effect by Segment" scatter shows every series lying on `y = x`. The page is plotting one variable as two. Searched all Group-3 tests — there is **no assertion** that `treatment_effect != uplift_score` (or that their correlation is `< 1.0`, or that the scatter is non-degenerate).
- **Sleeping dogs (n=857)** are correctly labelled and shown as a 4th segment on the by-segment bar (avg uplift -0.099). Exclusion from coupon eligibility is enforced by `tests/test_recommendations.py::test_negative_uplift_gets_no_action` and the LP-budget tests (`test_sleeping_dogs_get_zero_allocation`, `test_sleeping_dogs_excluded_from_objective`) — but those tests live in **other groups** (G2 budget / G4 recommendations), not Group 3.

### SaaS-readiness verdict — Experimentation domain

**Verdict:** **DO-NOT-SHIP**

**Rationale:**
1. **Test infrastructure is RED on Windows** — 161/381 tests cannot even reach an assertion because of a 1-line encoding bug. CI on any non-Korean-locale Windows host will fail the entire experimentation domain on file open. This is a portability blocker before any feature judgment.
2. **Page 11 ships a false-equivalence visualization** — `treatment_effect` and `uplift_score` are plotted as two independent quantities while being numerically identical to 4 decimal places. A SaaS customer will lose trust the moment they notice the y=x scatter.
3. **Page 06 has no empty-state UX** — zero experiments rendered as zero KPIs is broken-looking; combined with the absent lift-magnitude disclaimer (33.8% historical vs 5–15% literature), the page is misleading-by-omission.
4. **Page 04 has only 2 cohorts** — insufficient for any cohort-retention narrative, and the headline KPI (41.4%) doesn't match what's drawn (≈90%).

**Top deployment blockers:**
1. Fix `open(CONFIG_PATH, "r")` → `open(CONFIG_PATH, "r", encoding="utf-8")` across all test fixtures (impacts 6 of 8 files in Group 3 alone). Until then the suite is unrunnable on default Windows.
2. Page 11 — either remove the duplicate Treatment-Effect KPI/scatter, or compute and display a genuinely independent treatment-effect estimate (e.g. per-segment ATE from the experimental control). Add a regression test: `assert not np.allclose(uplift_score, treatment_effect, atol=1e-6)` (or `corr < 0.999`).
3. Page 06 — replace zero-state KPI cards with a single empty-state callout ("No experiments logged yet — see X to launch one") and surface the simulator-vs-literature lift disclaimer next to any future computed lift number.
4. Page 04 — generate ≥ 6 monthly cohorts (config or simulator change) and reconcile the "Avg Final Retention" KPI definition with the chart's terminal value (or rename the KPI to clarify the horizon).

**Test-coverage gaps you found:**
- **No test asserting `treatment_effect != uplift_score`** anywhere in the experimentation suites (or anywhere else — full-tree grep for `not.*allclose.*uplift|treatment_effect.*!=` confirms zero hits). The y=x bug is invisible to CI.
- **No uplift uniqueness floor test** (`>= 1000 unique values` per the audit ask). The current `np.std > 0.01` variance gate would still pass if uplift collapsed to e.g. ~100 binned values with sufficient spread; a strict `nunique >= 1000` (or `>= 0.2 * n`) assertion would harden the regression.
- **No cohort KPI/chart consistency test**. `test_cohort_analysis.py` (87 tests, all passing) validates the analytic functions but never compares a rendered KPI value to its companion chart's terminal value.
- **No A/B empty-state test** asserting the dashboard renders an empty-state message (vs all-zero KPIs) when `len(experiments) == 0`.
- **No A/B lift-realism guardrail** asserting `observed_lift <= 0.15` or surfacing a warning when synthetic lift exceeds the 5–15% literature band.
- **No sleeping-dog coupon-exclusion test in Group-3 surface** — coverage exists in Groups 2/4 but not co-located with the uplift suite that owns the segmentation.

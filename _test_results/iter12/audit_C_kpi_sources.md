# C — Dashboard KPI Source Verification

Audit scope: every headline KPI card rendered on the 16 iter11 dashboard pages
(`_test_results/dashboard_pages/*.png`). KPI list is taken from the iter9 page
dumps (`_test_results/iter9/page_data/*.md`) and verified against the live
dashboard render code in `src/dashboard/app.py`,
`src/dashboard/recommendations_view.py`, `src/dashboard/monitoring_view.py`,
`src/dashboard/system_health_view.py`, and `src/dashboard/data_loader.py`.

Source-type legend
- **REAL_ARTIFACT** — KPI is read directly from a pipeline-emitted file in
  `results/` (e.g. a single number lifted from `*.json`).
- **DERIVED_FROM_REAL** — KPI is computed in Python (`mean`, `sum`, `count`,
  threshold filter, …) from a DataFrame that itself was loaded from a real
  pipeline artifact in `results/`. The KPI is therefore real but is one
  transformation away from the on-disk number.
- **DERIVED_FROM_FALLBACK_SAMPLE** — KPI is computed from a DataFrame that the
  loader silently generates via `_generate_sample_*` because the expected
  pipeline artifact is missing from `results/` (and the loader does not refuse
  to render).
- **HARDCODED_FIXTURE** — KPI is a static Python literal in the codebase that
  is shown to the user with no I/O.
- **CONFIG** — value is read from `simulator_config.yaml`, not from a model
  output. (Counted as "real" because it is a real configuration value, not a
  random sample.)

Cross-cut notes (apply to every page)
- `results/clv_data.csv`, `clv_predictions.csv`, `churn_predictions.csv`,
  `model_metrics.json`, `feature_importance.csv`, `segments_6plus.csv`,
  `budget_results.csv`, `uplift_results.csv`, `recommendations.csv`,
  `monitoring_report.json`, `model_performance_history.csv`,
  `cohort_retention_matrix.csv`, `ab_test_detailed.json`,
  `ab_test_results.json`, `survival_results.json`, `cohort_analysis.json`,
  `budget_optimization_summary.json` — **EXIST** on disk.
- **NOT** present on disk: `confusion_matrices.json`, `roc_data.json`,
  `survival_curves.json`, `survival_data.csv`, `scoring_history.csv`,
  `scoring_throughput.csv`, `retention_offers.csv`, `drift_history.csv`.
  Loaders for these silently fall back to `_generate_sample_*` per
  `src/dashboard/data_loader.py` (`load_roc_data`, `load_confusion_matrices`,
  `load_survival_curves`, `load_survival_data` →
  `_load_survival_from_segments` is preferred, else sample;
  `load_scoring_history`, `load_scoring_throughput`, `load_retention_offers`).
- `load_drift_history` first tries `drift_history.csv` (missing), then
  synthesises a **1-row** DataFrame from `monitoring_report.json` — that single
  row is real, but every "trend" chart that uses it is a single point dressed
  as a time series.

---

## Per-page KPI source table

### Page 00 — Overview
Render path: `src/dashboard/app.py::render_overview` (lines 200-398).
Data source: `data_loader.load_predictions()` →
`_required_csv("churn_predictions.csv")` → `_adapt_predictions`.

| KPI | Value (iter9) | Source code path | Source type | Audit verdict |
|---|---|---|---|---|
| Total Customers | 20,000 | `len(predictions)` (app.py:229) | DERIVED_FROM_REAL (churn_predictions.csv) | ✅ |
| Avg Churn Prob | 31.31% | `predictions["churn_probability"].mean()` (app.py:230) | DERIVED_FROM_REAL | ✅ |
| High Risk | 5,717 | `(churn_probability > 0.5).sum()` (app.py:231) | DERIVED_FROM_REAL | ✅ |
| Total CLV | ₩57.94B | `predictions["clv_predicted"].sum()` then `format_currency_krw` (app.py:232, 239-246). `clv_predicted` is merged in via `_adapt_predictions` ← `load_clv_data()` ← `clv_data.csv` | DERIVED_FROM_REAL (clv_data.csv merged onto churn_predictions.csv) | ✅ |
| Customer C000000 Churn Probability | 3.09% | row lookup on `predictions` (app.py:363) | REAL_ARTIFACT (churn_predictions.csv row) | ✅ |
| Customer C000000 Risk Level | LOW | `row.get("risk_level", …)` (app.py:364) | REAL_ARTIFACT (column from churn_predictions.csv) | ✅ |
| Customer C000000 Segment | bargain_hunter | `row.get("segment", "unknown")` (app.py:365) | REAL_ARTIFACT | ✅ |
| Customer C000000 Predicted CLV | ₩2,716,186 | `row.get("clv_predicted", 0)` (app.py:369) | REAL_ARTIFACT (clv_data.csv) | ✅ |
| Customer C000000 Recommended Action | N/A | `row.get("recommended_action", "N/A")` — column not present in churn_predictions.csv | FALLBACK ("N/A" literal) | ⚠ value reflects that the field is not in the artifact, not a bug |
| Customer C000000 Days Since Purchase | 0 | `row.get("days_since_last_purchase", 0)` — column not present in churn_predictions.csv | FALLBACK (`.get` default 0) | ⚠ default 0 leaks as a real "0 days" reading |

### Page 01 — Churn Analytics
Render path: `app.py::render_churn_analytics` (3513+). Data:
`load_predictions()`, `load_feature_importance()`, `load_model_metrics()`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Customers | 20,000 | `len(predictions)` (app.py:3552) | DERIVED_FROM_REAL | ✅ |
| Avg Churn Prob | 31.31% | `predictions["churn_probability"].mean()` (app.py:3553) | DERIVED_FROM_REAL | ✅ |
| Median Churn Prob | 15.39% | `predictions["churn_probability"].median()` (app.py:3554) | DERIVED_FROM_REAL | ✅ |
| High Risk (>50%) | 5,717 | `(churn_probability > 0.5).sum()` (app.py:3555) | DERIVED_FROM_REAL | ✅ |
| Critical (>75%) | 3,596 | `(churn_probability > 0.75).sum()` (app.py:3556) | DERIVED_FROM_REAL | ✅ |
| ml_model AUC | 0.8852 | `model_metrics["ml_model"]["auc"]` ← `model_metrics.json::ml_metrics.auc_roc` | REAL_ARTIFACT | ✅ |
| ml_model Precision | 0.5331 | same payload | REAL_ARTIFACT | ✅ |
| ml_model Recall | 0.7791 | same payload | REAL_ARTIFACT | ✅ |
| ml_model F1 | 0.6331 | same payload | REAL_ARTIFACT | ✅ |
| dl_model AUC | 0.8860 | `model_metrics["dl_model"]["auc"]` | REAL_ARTIFACT | ✅ |
| dl_model P/R/F1 | 0.6759 / 0.6318 / 0.6531 | same payload | REAL_ARTIFACT | ✅ |
| ensemble AUC | 0.8866 | same payload | REAL_ARTIFACT | ✅ |
| ensemble P/R/F1 | 0.6426 / 0.6621 / 0.6522 | same payload | REAL_ARTIFACT | ✅ |
| At-Risk Revenue banner | ₩2,997,471,916 / 5.2% | `predictions[churn>0.5]["clv_predicted"].sum()`, ratio with total | DERIVED_FROM_REAL | ✅ |

### Page 02 — Model Performance
Render path: `app.py::render_model_performance` (406-). Data:
`load_model_metrics()` and `load_confusion_matrices()`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| ML Model AUC | 0.8852 | `model_metrics.json` | REAL_ARTIFACT | ✅ |
| DL Model AUC | 0.8860 | `model_metrics.json` | REAL_ARTIFACT | ✅ |
| Ensemble AUC | 0.8866 | `model_metrics.json` | REAL_ARTIFACT | ✅ |
| Best Model | ensemble | `max(metrics.items(), key=lambda x: x[1]["auc"])` | DERIVED_FROM_REAL | ✅ |
| Performance Comparison table | full P/R/F1 grid | `pd.DataFrame(metrics).T` but **then overwritten** by confusion-matrix recompute (app.py:439-463) | MIXED: AUC real, P/R/F1 derived from `_generate_sample_confusion_matrices()` (FALLBACK_SAMPLE) because `confusion_matrices.json` does not exist | ❌ — see "Top fishy KPIs" |
| ROC AUCs (0.885 / 0.886 / 0.887) | chart labels | `_generate_sample_roc_data()` because `roc_data.json` is missing; AUCs are scaled to match the real model_metrics AUCs, but FPR/TPR points are synthetic | DERIVED_FROM_FALLBACK_SAMPLE | ⚠ visually plausible, mathematically fake |
| Confusion matrices (350/50/80/120 etc.) | hardcoded literals in `_generate_sample_confusion_matrices` (data_loader.py:1348-1354) | HARDCODED_FIXTURE | ❌ |

### Page 03 — Customer Segmentation
Render path: `app.py::render_segmentation` (~768+). Data: `load_predictions()`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Segments | 6 | `predictions["segment"].nunique()` (app.py:792) | DERIVED_FROM_REAL (segment column from churn_predictions.csv) | ✅ |
| Total Customers | 20,000 | `len(predictions)` (app.py:793) | DERIVED_FROM_REAL | ✅ |
| Highest Risk Segment | dormant | `groupby("segment")["churn_probability"].mean().idxmax()` (app.py:794-798) | DERIVED_FROM_REAL | ✅ |
| Per-segment counts (donut + bar) | 2,030 vip_loyal etc. | `value_counts()` on predictions | DERIVED_FROM_REAL | ✅ |
| Avg CLV by segment | per-segment CLV | groupby on `clv_predicted` merged from clv_data.csv | DERIVED_FROM_REAL | ✅ |

### Page 04 — Cohort Analysis
Render path: `app.py::render_cohort_analysis` (~3865+). Data:
`load_cohort_retention_matrix()` → `cohort_retention_matrix.csv`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Cohorts | 4 | `retention_matrix.shape[0]` (app.py:3965) | DERIVED_FROM_REAL | ✅ |
| Periods Tracked | 13 | `retention_matrix.shape[1]` | DERIVED_FROM_REAL | ✅ |
| Avg Period-1 Retention | 99.0% | `retention_matrix[1].mean()` | DERIVED_FROM_REAL | ✅ |
| Avg Final / Deepest-Observed Retention | 2.5% | mean of `retention_matrix[last_col]` over observed cells (app.py:3946-3963). iter11 added unobserved-cell masking | DERIVED_FROM_REAL | ✅ |
| Retention heatmap matrix (Jan/Feb/Mar/Apr 2024 × 13 periods) | full grid | `cohort_retention_matrix.csv` cell values | REAL_ARTIFACT | ✅ |
| "Period 12 = 10.2%" etc. | last-period values | matrix lookup | REAL_ARTIFACT (the truncation artifact is real, just data quality) | ⚠ data quality, not source fakery |

### Page 05 — Budget Optimization
Render path: `app.py::render_budget` (~1000-1500). Data:
`load_budget_results()` → `budget_results.csv`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Allocated | ₩50,000,000 | `display_results["allocated_budget_krw"].sum()` (app.py:1090) — after scaling by config-driven slider, but baseline is from budget_results.csv | DERIVED_FROM_REAL | ✅ |
| Expected Retained | 122 (iter11) / 118 (iter9) | `int((budget_results["expected_retained"] * scale * uplift).sum())` (app.py:1099-1101) | DERIVED_FROM_REAL | ✅ |
| Revenue Saved | ₩192,155,551 | `display_results["expected_revenue_saved_krw"].sum()` | DERIVED_FROM_REAL | ✅ |
| ROI (budget envelope) | 3.84x (iter11) / 3.5x (iter9 mean-of-segments) | `compute_overall_roi(total_rev_saved, total_alloc, scope="budget")` (app.py:1106-1110) | DERIVED_FROM_REAL | ✅ iter11 fix |
| Per-segment allocations & ROIs | 8 segments | rows of `budget_results.csv` | REAL_ARTIFACT | ✅ |
| What-If scenario rows (Baseline, Conservative -30%, Aggressive +50%, …) | retained 122/68/220/… | `_compute_scenario_comparison` recomputes by scaling baseline; underlying retained/ROI numbers traced to `budget_results.csv` rows scaled by simple multipliers in code | DERIVED_FROM_REAL (with deterministic in-app scaling) | ⚠ scaling factors (-30% / +50% / cost_reduction) are hardcoded model assumptions, not from artifact |

### Page 06 — A/B Testing
Render path: `app.py::render_ab_testing` (1428-). Data:
`load_ab_test_detailed()` → `ab_test_detailed.json` (now present) / fallback to
`ab_test_results.json`.

| KPI | Value (iter9 dump) | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Experiments | 0 (iter9) → now 2 in iter11 (ab_test_detailed.json has 2 experiments) | `summary.get("total_experiments", len(experiments))` (app.py:1461) | REAL_ARTIFACT (when artifact present) — iter9 was an empty-summary state | ✅ |
| Significant Results | 0 → 2 | `summary.get("significant_count", 0)` | REAL_ARTIFACT | ✅ |
| Best Experiment | N/A → "simulated_retention_campaign" | `summary.get("best_experiment", "N/A")` | REAL_ARTIFACT | ✅ |
| Avg Lift | 0.0% → ~26% | `summary.get("avg_lift", 0)` | REAL_ARTIFACT | ✅ |
| Required Sample Size (per group) | 906 | computed live by `compute_sample_size_two_proportions` from slider inputs (baseline=0.20, MDE=0.05, α=0.05, power=0.8); not artifact-driven | DERIVED (live computation from CONFIG/slider inputs, not from a model artifact) | ✅ correct math |
| Total Participants Needed | 1,812 | `2 × required_per_group` | DERIVED | ✅ |
| Expected Duration (days) | 19 | derived from sample size and assumed daily traffic in app.py | DERIVED (with hardcoded traffic assumption) | ⚠ |

### Page 07 — Survival Analysis
Render path: `app.py::render_survival` (~1850-). Data: `load_survival_data()`
(prefers `survival_data.csv` — **missing** — falls back to
`_load_survival_from_segments` which **derives a fake survival table from
`segments_6plus.csv`** by `duration = 365 × (1 - churn_probability)` and
`event_observed = (churn ≥ 0.5)`); plus `load_survival_curves()` which always
falls back to `_generate_sample_survival_curves` because
`survival_curves.json` is missing.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Customers | 20,000 | `len(survival)` (app.py:1891) | DERIVED_FROM_REAL (rows in segments_6plus.csv) | ✅ |
| Predicted Churners (>50%) | 5,717 | `survival["event_observed"].sum()` (app.py:1892). iter11 renamed from "Events Observed" — `event_observed` is `(churn_probability ≥ 0.5).astype(int)` in `_load_survival_from_segments` (data_loader.py:457). So this **equals** the High-Risk count on Page 01 by construction | DERIVED_FROM_REAL but the column is a recoded prediction, NOT a survival event | ⚠ iter11 added a help-tooltip explaining this; verdict: surfaced but still tautological |
| Predicted Churn Rate | 28.59% | `event_count / total_cust` | DERIVED_FROM_REAL (same caveat) | ⚠ |
| Median Duration | 309 days | `survival["duration_days"].median()` where `duration = 365×(1-churn_prob)` — a synthetic transform, not a real time-to-event | DERIVED but the underlying duration column is a deterministic function of churn probability, not a real survival duration | ❌ — see "Top fishy KPIs" |
| Avg Survival Probability by uplift segment | 97.68%/86.98%/… | `survival.groupby("segment")["survival_probability"].mean()`. `survival_probability = 1 - churn_probability` in `_load_survival_from_segments` | DERIVED from churn predictions, not a Cox PH model output | ❌ — KM curves likewise come from `_generate_sample_survival_curves` (8 hardcoded base survivals; 360-day decay constants) |
| Daily Hazard Rate by behavioral segment | 0.00254 dormant etc. | derived from sample KM curves `(-log(s_end/s0)/t_max)` (app.py:2083) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Event Rate by uplift segment (0% / 100% binary) | per uplift segment | `survival.groupby("segment")["event_observed"].mean()` where uplift segments are defined post-hoc using the same churn outcome | DERIVED_FROM_REAL but tautological by construction | ⚠ iter11 added a warning banner |
| Survival Model Config (penalizer 0.01, l1_ratio 0, alpha 0.05) | static JSON | hardcoded display | CONFIG (display only) | ✅ |

### Page 08 — Model Monitoring
Render path: `src/dashboard/monitoring_view.py::render_model_monitoring`.
Data: `load_drift_history()` (1-row from `monitoring_report.json`),
`load_scoring_throughput()` (sample), `load_model_metrics()` (real),
`load_model_performance_history()` (real,
`model_performance_history.csv`).

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total (Drift) Checks | 1 | `len(drift_history)` (monitoring_view.py:150) | DERIVED_FROM_REAL but n=1 because the loader synthesises one row from `monitoring_report.json` | ⚠ |
| Current Status | RED | `drift_history.iloc[-1]["alert_level"]` ← `monitoring_report.json::overall_alert_level` | REAL_ARTIFACT | ✅ |
| Red Alerts | 1 | `(drift_history["alert_level"] == "red").sum()` | DERIVED_FROM_REAL (over 1 row) | ⚠ |
| Yellow Alerts | 0 | same | DERIVED_FROM_REAL (over 1 row) | ⚠ |
| Avg Requests/min | 49.0 | `scoring_throughput["requests_per_minute"].mean()` from `_generate_sample_scoring_throughput` (sinusoidal pattern, Oct 15 2024 timestamps) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Peak Requests/min | 83.3 | `.max()` on same sample | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Avg Latency | 19.1 ms | `.mean()` on sample (`15 + Expo(5)`) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Avg Error Rate | 0.0103 / 1.03% | `.mean()` on sample (`Uniform(0, 0.02)`) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |

### Page 09 — Recommendations
Render path: `src/dashboard/recommendations_view.py::render_recommendations_view`.
Data: `load_recommendations()` → `recommendations.csv`,
`load_retention_offers()` (fallback sample), `load_predictions()`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Recommendations | 20,000 | `len(recs)` (recommendations_view.py:207) | DERIVED_FROM_REAL | ✅ |
| Avg Predicted Uplift (all customers) | 6.36% | `recs["expected_uplift"].mean()` (recommendations_view.py:211) | DERIVED_FROM_REAL | ✅ |
| Top Action Type | no_action | `recs["recommendation_type"].value_counts().idxmax()` | DERIVED_FROM_REAL | ✅ |
| High Priority | 16,106 | `(recs["priority_score"] >= 0.7).sum()` | DERIVED_FROM_REAL | ✅ |
| Total Campaign Cost | ₩1,211,055 | `retention_offers["estimated_cost_krw"].sum()` ← **`_generate_sample_retention_offers`** (n=50 sample with random costs) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Est. Revenue Saved | ₩10,893,463 | same sample | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Overall ROI (cost-benefit strip) | 9.0x | `revenue/cost` on sample | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Avg Expected Uplift (cost-benefit strip) | 10.88% | mean of `retention_offers["expected_uplift"]` sample | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Action-type distribution (no_action 83% / coupon 17%) | from recommendations.csv | DERIVED_FROM_REAL | ✅ |
| Cost-by-offer-type (premium_discount ₩559,170 etc.) | from sample retention_offers | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |

### Page 10 — CLV Prediction
Render path: `app.py::render_clv` (2413-). Data: `load_predictions()` +
`load_clv_data()` (clv_data.csv) merged.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total CLV | ₩57,936,514,970 | `predictions["clv_predicted"].sum()` (app.py:2474) | DERIVED_FROM_REAL (clv_data.csv) | ✅ |
| Average CLV | ₩2,896,826 | `.mean()` | DERIVED_FROM_REAL | ✅ |
| Median CLV | ₩1,701,727 | `.median()` | DERIVED_FROM_REAL | ✅ |
| CLV Std Dev | ₩3,575,497 | `.std()` | DERIVED_FROM_REAL | ✅ |
| Per-segment Mean/Total CLV (8 uplift segments) | full table | groupby on real clv_data merged with segments_6plus.csv | DERIVED_FROM_REAL | ✅ |
| CLV Percentiles P10..P99 | per percentile | `predictions["clv_predicted"].quantile([0.1, …])` | DERIVED_FROM_REAL | ✅ |
| CLV Tier Distribution (Platinum/Gold/Silver/Bronze all 25%) | quartile cut | DERIVED_FROM_REAL (`pd.qcut` on clv_data); tier shares are always 25% by construction | ⚠ trivial-by-design |

### Page 11 — Uplift Modeling
Render path: `app.py::render_uplift` (2762-). Data: `load_uplift_results()` →
`uplift_results.csv`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Avg Uplift Score | 0.0434 | `uplift["uplift_score"].mean()` (app.py:2791) | DERIVED_FROM_REAL | ✅ |
| Persuadable | 16,317 (iter11 KPI broken into 4 quadrants) | `uplift["segment"].value_counts()["persuadable"]` from uplift_results.csv. iter11 expanded the strip to 5 cards (Avg Uplift + 4 quadrant counts) | DERIVED_FROM_REAL | ✅ |
| Sure Thing | (iter11 added) | same | DERIVED_FROM_REAL | ✅ |
| Sleeping Dog | 3,683 | same | DERIVED_FROM_REAL | ✅ |
| Lost Cause | (iter11 added) | same | DERIVED_FROM_REAL | ✅ |
| Avg Treatment Effect (iter9 dump, now removed) | 0.0434 — equal to Avg Uplift | iter11 removed this card; the underlying `uplift_results.csv` has `treatment_effect == uplift_score` on every row, so the equality was structural, not a display bug | REAL_ARTIFACT but the second column is a duplicate of the first in the artifact itself | ⚠ data issue surfaced inline via a caption |
| Per-segment Avg Uplift (4-quadrant bar) | 0.19/0.06/0.03/-0.11 | groupby on uplift_results.csv | DERIVED_FROM_REAL | ✅ |

### Page 12 — CLV & Retention Campaign
Render path: `app.py::render_retention_campaign`. Data:
`load_predictions()`, `load_clv_data()`, `load_uplift_results()`,
`load_budget_results()`.

| Section | KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|---|
| 1 CLV | Total CLV | ₩57.94B | `predictions["clv_predicted"].sum()` (app.py:3091) | DERIVED_FROM_REAL | ✅ |
| 1 CLV | Avg CLV | ₩2,896,826 | `.mean()` | DERIVED_FROM_REAL | ✅ |
| 1 CLV | At-Risk CLV | ₩2,997,471,916 | `predictions[churn>0.5]["clv_predicted"].sum()` | DERIVED_FROM_REAL | ✅ |
| 1 CLV | At-Risk CLV % | 5.2% | `at_risk / total × 100` | DERIVED_FROM_REAL | ✅ |
| 2 Uplift | Avg Uplift | 0.0434 | uplift_results.csv mean | DERIVED_FROM_REAL | ✅ |
| 2 Uplift | Max Uplift | 0.6874 | `.max()` | DERIVED_FROM_REAL | ✅ |
| 2 Uplift | Treatable Customers | 16,317 (81.6%) | `(uplift_score > 0).sum()` | DERIVED_FROM_REAL | ✅ |
| 3 Budget | Budget Allocated | ₩50,000,000 | `budget_data["allocated_budget_krw"].sum()` | DERIVED_FROM_REAL | ✅ |
| 3 Budget | Revenue Saved | ₩192,155,554 | `expected_revenue_saved_krw.sum()` | DERIVED_FROM_REAL | ✅ |
| 3 Budget | Customers Retained | 122 (iter9 dump 122.295…) | iter9 had the float leak; iter11 fixed via `format_count(total_retained, integer=True)` (app.py:3257) | DERIVED_FROM_REAL | ✅ |
| 3 Budget | Overall ROI | 3.8x (iter11) | `compute_overall_roi(scope="budget")` | DERIVED_FROM_REAL | ✅ |
| Section 4 ROI summary rows | various | computed from budget_data | DERIVED_FROM_REAL | ✅ |

### Page 13 — Real-Time Scoring (3 tabs)
Render path: `app.py::_render_live_scoring_tab` / `_render_retention_offers_tab`
/ `_render_monitoring_tab`. Data: live Redis probe + `load_scoring_history()`
(sample), `load_scoring_throughput()` (sample), `load_retention_offers()`
(sample), `load_drift_history()` (1-row from monitoring_report.json).

#### Tab a — Live Scoring Status
| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Request queue depth (iter11) / Request Stream (iter9) | 0 | `r.xlen("scoring_requests")` live Redis | REAL (live probe) | ✅ |
| Response queue depth | 0 | `r.xlen("scoring_responses")` | REAL (live probe) | ✅ |
| Consumer Group | scoring_consumers | `redis_config["consumer_group"]` | CONFIG | ✅ |
| Total Scores (lifetime) | 200 | `len(scoring_history)` ← **`_generate_sample_scoring_history`** n=200 (data_loader.py:1749-1780) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Avg Churn Prob | 27.30% | `scoring_history["churn_probability"].mean()` of sample (`np.random.beta(2,5,200)`) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| High/Critical Risk | 17 | filter on sample risk_level | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Primary Model | ensemble | `scoring_history["model_type"].value_counts().index[0]` of sample (sampled with p=[0.7,0.15,0.15]) | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |

#### Tab b — Retention Offer Recommendations
| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Offers | 44 (/200 if denom available) | `len(filtered)` on `retention_offers.csv` ← fallback sample (n=50) post-filter | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Total Cost | ₩1,196,659 | `filtered["estimated_cost_krw"].sum()` of sample | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Expected Revenue Saved | ₩10,752,341 | `filtered["estimated_revenue_save_krw"].sum()` of sample | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Expected ROI (treated) | 8.99x (iter11) / 8.0x (iter9) | `compute_overall_roi(scope="treated")` on the sample numerator/denominator | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Quick-lookup Recommendation / Expected Uplift / Risk Score | from recommendations.csv | REAL_ARTIFACT (recommendations.csv) | ✅ (per-customer lookup is real) |

#### Tab c — Model Monitoring
| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Drift Checks | 1 | `len(drift_history)` (one row synth'd from `monitoring_report.json`) | DERIVED_FROM_REAL but n=1 | ⚠ |
| Red Alerts | 1 | `(alert_level == "red").sum()` | DERIVED_FROM_REAL (n=1) | ⚠ |
| Yellow Warnings | 0 | same | DERIVED_FROM_REAL (n=1) | ⚠ |
| Latest Alert Level | RED | `drift_history.iloc[-1]["alert_level"]` ← `monitoring_report.json::overall_alert_level` | REAL_ARTIFACT | ✅ |

### Page 14 — MLflow Experiments
Render path: `app.py::render_mlflow_experiments` (~5230+). Data:
`_probe_mlflow_status()` (live TCP) + `load_mlflow_runs()` →
`load_model_performance_history()` → `model_performance_history.csv`.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| Total Runs | 3 | `len(mlflow_runs)` (app.py:5304) ← `model_performance_history.csv` (3 rows: ml/dl/ensemble) | DERIVED_FROM_REAL | ✅ |
| Best AUC | 0.8866 | `mlflow_runs.loc[mlflow_runs["auc"].idxmax(), "auc"]` | DERIVED_FROM_REAL | ✅ |
| Best Model | ensemble | same row | DERIVED_FROM_REAL | ✅ |
| Total Training Time | 3s | `mlflow_runs["training_time_s"].sum()` — `_load_model_performance_history` defaults `training_time_s = 1.0` per row when the artifact column is absent (data_loader.py:1162) | DERIVED but **the per-row training_time_s is a hardcoded default 1.0** when the CSV does not log it | ⚠ — see "Top fishy KPIs" |
| AUC values per model (chart) | 0.8852/0.8860/0.8866 | model_performance_history.csv | REAL_ARTIFACT | ✅ |
| Learning Rate column (params_lr = 0.1) | column constant | `_load_model_performance_history` defaults `params_lr = 0.1` when missing (data_loader.py:1168) | HARDCODED_FIXTURE (column default) | ❌ — iter11 already gates the "Learning Rate vs AUC" chart on degeneracy and emits a caption |
| Epochs column (params_epochs = 1) | column constant | same fallback (data_loader.py:1170) | HARDCODED_FIXTURE | ❌ |

### Page 15 — System Health
Render path: `src/dashboard/system_health_view.py`. Data: live
`check_redis_health()`, `check_mlflow_health()`, file-system scan of
`results/` for pipeline artifacts.

| KPI | Value | Source code path | Source type | Verdict |
|---|---|---|---|---|
| 3/3 services healthy | header | live probes (`check_redis_health`, `check_mlflow_health`, pipeline scan) | REAL (live probes) | ✅ |
| Redis Connected | Yes | `redis_health["connected"]` via `r.ping()` | REAL (live probe) | ✅ |
| Redis Stream (requests) / (responses) | 0 / 0 | `r.xlen(...)` live | REAL (live probe) | ✅ |
| MLflow Connected | Yes / cached | iter11 reconciles via `mlflow_runs` fallback when API empty (system_health_view.py:692-718) | REAL (live probe with cached-runs fallback) | ✅ iter11 fix |
| MLflow Experiments | 0 → max(live, runs_experiment_count) | iter11 takes `max(exp_from_health, runs_experiment_count)` so 0/3 contradiction is gone | DERIVED | ✅ iter11 fix |
| Pipeline Artifacts | 44 | `len(list(results_dir.glob("*")))` | DERIVED_FROM_REAL (filesystem) | ✅ |
| Pipeline Models | 4 | count of model files | DERIVED_FROM_REAL | ✅ |
| Avg Throughput / Latency / Error Rate (49 / 19.1 ms / 0.0103) | reused from Page 08 | same `_generate_sample_scoring_throughput` sample | DERIVED_FROM_FALLBACK_SAMPLE | ❌ |
| Total Runs / Best AUC / Best Model / Total Train Time (3 / 0.8866 / ensemble / 3s) | reused from Page 14 | `model_performance_history.csv` (training_time_s default 1.0) | DERIVED_FROM_REAL but Total Train Time defaults to 1s/row | ⚠ |

---

## Aggregate statistics

Counts per KPI (headline strip + secondary KPI cards explicitly enumerated in
the iter9 page dumps, excluding chart-only values).

| Bucket | Count | % |
|---|---:|---:|
| REAL_ARTIFACT (1-to-1 from a results file) | 21 | 19.4% |
| DERIVED_FROM_REAL (computed in code from a real artifact) | 49 | 45.4% |
| CONFIG (real configuration value) | 3 | 2.8% |
| Real (live system probe — Redis, MLflow API, FS scan) | 5 | 4.6% |
| **Subtotal — real or derived from real** | **78** | **72.2%** |
| DERIVED_FROM_FALLBACK_SAMPLE | 23 | 21.3% |
| HARDCODED_FIXTURE (literal constants) | 5 | 4.6% |
| FALLBACK / default leak (e.g. `.get(col, 0)`) | 2 | 1.9% |
| UNCLEAR | 0 | 0% |
| **Total KPIs audited** | **108** | **100%** |

Rough headline numbers requested by the brief:
- **Total KPIs audited:** 108 (across 16 pages + Page 13's 3 tabs).
- **Real or derived-from-real:** 78 / 108 = **72.2%**.
- **Hardcoded / fallback-sample / default-leak:** 30 / 108 = **27.8%**.
- **Unclear:** 0.

---

## Top fishy KPIs (suspected hardcode / fixture)

1. **Page 02 — Confusion-matrix-derived Precision/Recall/F1/Accuracy on the
   headline KPI strip & in the Performance Comparison table.**
   Source: `_generate_sample_confusion_matrices` (`data_loader.py:1348-1354`)
   returns the hardcoded 3×2×2 matrices `{ml:[[350,50],[80,120]], dl:[[340,60],
   [90,110]], ensemble:[[360,40],[70,130]]}` because `confusion_matrices.json`
   is **not** generated by the pipeline. `render_model_performance` (lines
   439-463) then **overwrites** the real `precision/recall/accuracy/f1` from
   `model_metrics.json` with values recomputed from these fake matrices. So
   the entire Performance Comparison table prints fixture-derived numbers
   while the headline AUCs stay real. This is the most load-bearing fake on
   the dashboard.

2. **Page 07 — Median Duration 309 days, Avg Survival Probability by segment,
   Daily Hazard Rate by segment.**
   `survival_data.csv` does not exist; `load_survival_data` falls back to
   `_load_survival_from_segments` which deterministically derives every
   "survival" column from `churn_probability`:
   `duration_days = 365 × (1 - churn_probability)`,
   `event_observed = (churn ≥ 0.5)`,
   `survival_probability = 1 - churn_probability`. Then
   `load_survival_curves` falls back to `_generate_sample_survival_curves`,
   which hardcodes 6 base-survival anchors `{vip_loyal:0.92, …, dormant:0.40}`
   and emits exponential decay curves. None of the KM curves, the hazard
   table, or the median-duration KPI come from a fitted Cox/KM model.

3. **Page 08 — Throughput / Latency / Error Rate KPIs (49.0 req/min, 19.1 ms,
   1.03%) and the entire Oct 15-16 2024 charts.**
   `scoring_throughput.csv` is missing; `_generate_sample_scoring_throughput`
   synthesises a 48-point sinusoidal pattern starting "2024-10-15" with
   `15 + Expo(5)` ms latency and `Uniform(0, 0.02)` error rate. The same KPIs
   are reused on Page 13 (tab a) and Page 15.

4. **Page 09 (cost-benefit strip) and Page 13 tab b — Total Campaign Cost
   ₩1,211,055, Est. Revenue Saved ₩10,893,463, Overall ROI 9.0x / 8.0x, plus
   per-offer-type cost & ROI breakdowns.**
   All derive from `_generate_sample_retention_offers` (n=50 sample), because
   `retention_offers.csv` does not exist. The numbers are correctly summed
   but the underlying offers were never produced by an optimizer — they are
   bucketed by risk/segment with hardcoded `np.random.uniform(...)` cost
   ranges.

5. **Page 13 tab a — Total Scores 200, Avg Churn Prob 27.30%, High/Critical
   Risk 17, Primary Model ensemble.**
   Driven by `_generate_sample_scoring_history` (n=200,
   `np.random.beta(2,5,200)`, timestamps starting `2024-10-01` at 15-min
   intervals, model sampled `p=[0.7,0.15,0.15]`). No `scoring_history.csv`
   exists. The Redis depth (0/0) next to "200" is the real-vs-fake split
   referenced in the iter9 audit.

Honorable mention:
- **Page 14 — Total Training Time 3s, params_lr 0.1, params_epochs 1.** The
  `model_performance_history.csv` artifact does **not** include
  `training_time_s`, `params_lr`, or `params_epochs` columns; the loader
  fills them with hardcoded defaults `1.0 / 0.1 / 1` (data_loader.py:1162,
  1168, 1170). The "Total Training Time = 3s" is literally `1.0 × 3 rows`.
  iter11 already detects the degenerate sweep and replaces the LR-vs-AUC
  scatter with a caption.

---

## Verified-real KPIs (high confidence)

These KPIs are read or computed directly from a pipeline-emitted artifact in
`results/` and trace to a verifiable artifact value:

- All Page 00, 01, 03, 10, 12 (CLV section) `Total Customers`, `Avg/Median
  Churn`, `High Risk`, `Critical`, `Total CLV`, `Avg CLV`, `Median CLV`,
  `CLV Std Dev`, `At-Risk CLV` — sourced from `churn_predictions.csv` and
  `clv_data.csv` via `load_predictions()` + `load_clv_data()`.
- Page 01, 02, 14 AUC/Precision/Recall/F1 **AUC ONLY** — from
  `model_metrics.json` and `model_performance_history.csv`. (Precision /
  Recall on Page 02 are overwritten with confusion-matrix-derived fakes.)
- Page 03 segment distribution, sample sizes, mean churn per segment, mean
  CLV per segment — groupby on `churn_predictions.csv` + clv_data.csv.
- Page 04 entire cohort retention matrix — `cohort_retention_matrix.csv`.
- Page 05 Budget Allocated / Revenue Saved / Expected Retained / ROI / per-
  segment breakdown — `budget_results.csv` and
  `budget_optimization_summary.json`.
- Page 06 experiment names, lifts, p-values, sample sizes — from
  `ab_test_detailed.json` (real pipeline output despite iter9 showing the
  empty-state version).
- Page 10 CLV distribution, percentiles, per-segment Mean/Total CLV — from
  `clv_data.csv`.
- Page 11 Avg Uplift / 4-quadrant counts / per-segment averages — from
  `uplift_results.csv`.
- Page 12 Section 3 budget envelope numbers — from `budget_results.csv`.
- Page 14 Total Runs (=3), AUC values per model, Best AUC, Best Model — from
  `model_performance_history.csv`.
- Page 15 service-health probes (Redis ping + xlen, MLflow API, filesystem
  scan of `results/`) — live and real.

---

## Recommendations

1. **Stop calling `_generate_sample_*` from production loaders.** Make
   `load_confusion_matrices`, `load_roc_data`, `load_survival_curves`,
   `load_scoring_history`, `load_scoring_throughput`,
   `load_retention_offers`, and `load_drift_history` fail to a clearly
   labelled empty state (the way `load_budget_results` and `load_predictions`
   already do via `_required_csv`) when their artifact is missing. The
   current behaviour silently injects randomly-generated KPIs onto Pages 02,
   07, 08, 09, 13, 15.

2. **Stop overwriting real metrics with fixture-derived ones.** Page 02
   `render_model_performance` (app.py:439-463) deliberately replaces
   `model_metrics.json` precision/recall/accuracy/f1 with values recomputed
   from `_generate_sample_confusion_matrices`. The original motivation was
   to reconcile the KPI strip with the confusion-matrix tile, but the
   correct fix is to either emit a real `confusion_matrices.json` from the
   pipeline or to hide the matrix tile when no such artifact exists — not to
   downgrade the real KPI strip.

3. **Page 07 survival data is purely re-encoded predictions.** The "Median
   Duration 309 days" headline is a deterministic function of
   `churn_probability`, not a fitted survival model. Either point the page
   at `survival_results.json` (which already has `median_survival_days =
   295.0`, `concordance_index = 0.821`, `num_events = 3999`) or load a real
   `survival_data.csv` produced by the survival pipeline. The current
   `_load_survival_from_segments` fallback is the highest-risk fake on the
   dashboard because it looks like a Cox PH output.

4. **Surface the synthetic-data tag inline.** iter11 already shows the
   "Synthetic data — FULL mode" banner at the top of each page. Extend the
   same labelling to **specific KPI cards** that are sample-fallback driven
   (Page 02 P/R/F1, Page 07 median duration, Page 08 throughput/latency,
   Page 09 mid-page cost-benefit strip, Page 13 tab a / tab b). The
   page-level banner can be read as "synthetic customer events", while the
   KPI cards in question are downstream-of-customer-events synthetic, which
   is a different and less-acceptable failure mode.

5. **Add an "artifact present?" badge to each page.** `DashboardDataLoader`
   already tracks `_artifact_issues`. Render a small green/red dot next to
   each KPI strip indicating whether the underlying artifact was found, so
   the operator can distinguish a real number from a sample-driven one
   without grepping the codebase.

6. **`model_performance_history.csv` should log `training_time_s`,
   `params_lr`, `params_epochs`.** All three are currently filled in with
   loader-side defaults (1.0, 0.1, 1), which is why Page 14 shows "Total
   Training Time = 3s" and a degenerate LR-vs-AUC sweep. The pipeline writer
   needs to emit these columns; the loader's `.fillna(1.0)` default is
   masking that.

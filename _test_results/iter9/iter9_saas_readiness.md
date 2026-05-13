# Iter 9 — Live Dashboard SaaS-Readiness Audit (full-content PNG + structured data, 6 zero-context agents)

**Date:** 2026-05-10
**Method:**
1. **Re-captured 18 PNGs** from the live Streamlit dashboard at `http://localhost:8501` (FULL mode, n=20,000) using Playwright with viewport 1600×6500 and inner-container scroll-trigger to force lazy Plotly renders → entire page (every chart, table, KPI) in each PNG.
2. **Extracted structured data per page** to `_test_results/page_data/*.md` — banners, headers, all KPI cards (label/value), all Plotly chart titles + axes, full innerText. 16 page-data MDs.
3. **Dispatched 6 sub-agents in parallel, each with NO prior project context** — each agent received its assigned PNGs (visual layout) + matching MDs (exact numbers).
4. Main orchestrator (this file) consolidates the 6 independent reports.

**Aggregate verdict: 6/6 agents → NOT SaaS-ready / DO-NOT-SHIP for paid pilot.**

---

## 0. Coverage map

| Agent | Pages | Verdict |
|---|---|:--:|
| A1 | 00 Overview · 01 Churn Analytics · 02 Model Performance | **DO-NOT-SHIP** |
| A2 | 03 Customer Segmentation · 04 Cohort Analysis · 05 Budget Optimization | **NOT SHIP-READY** |
| A3 | 06 A/B Testing · 07 Survival Analysis · 11 Uplift Modeling | **NOT SAAS-READY** |
| A4 | 08 Model Monitoring · 14 MLflow Experiments · 15 System Health | **NOT SAAS-READY** |
| A5 | 09 Recommendations · 10 CLV Prediction · 12 CLV & Retention Campaign | **NOT SAAS-READY** |
| A6 | 13 Real-Time Scoring (3 tabs) | **NOT PRODUCTION-READY** |

(15 distinct pages + 3 tabs of page 13 = 18 PNGs; all freshly captured.)

---

## 1. Same-page contradictions (most damaging — defeats trust before procurement asks)

These are P0 because the dashboard contradicts ITSELF on the same page or across two pages of the same product. A single screenshot is enough to defeat it.

| # | Where | Defect | Source |
|---|---|---|---|
| 1 | **Page 02** | Headline KPI table: ml_model **Precision 0.5331 / Recall 0.7791**. Confusion matrix below it: **Precision 0.7059 / Recall 0.6000**. Δ +17.3pt precision, -17.9pt recall on the same model. Ensemble has +12.2pt precision gap. **At least one is wrong.** | A1 |
| 2 | **Page 02** | Confusion matrices total **600 cases**, headline AUCs claim to be on a 20,000-customer test set. Undisclosed 3% subsample backing the model-performance numbers. | A1 |
| 3 | **Page 14 vs Page 15** | Page 14 banner: *"MLflow tracking server not available — showing cached experiment data."* Page 15 KPI: *"MLflow Tracking — Connected: Yes."* Two pages, opposite states. | A4 |
| 4 | **Page 15** | Same page: KPI says **Experiments = 0** AND **Total Runs = 3**. Same page: header **"All Systems Operational ✅"** AND **"Current Drift Status: RED"**. | A4 |
| 5 | **Page 08** | Banner: **"No performance degradation detected for ensemble"**. KPI card directly above: **"Current Status: RED, Red Alerts: 1"**. | A4 |
| 6 | **Page 11** | Headline: **Persuadable 16,317 + Sleeping Dogs 3,683 = 20,000**. 4-segment table on same page: persuadable=2,708, sure_thing=12,929, sleeping_dog=3,683, lost_cause=600 = 19,920. Pie legend: "Persuadable / **Lost Cause**". **Three different vocabularies on one page.** | A3 |
| 7 | **Page 11** | Avg Uplift Score = **0.0434**, Avg Treatment Effect = **0.0434** (4-decimal match). uplift_score == treatment_effect on every Top-10 row. **One variable plotted as two metrics.** | A3 |
| 8 | **Page 13 tab a** | Request Stream **0** + Response Stream **0** + Total Scores **200**. Either streams are dead and chart is fake, or chart is real and KPI cards are dead. | A6 |
| 9 | **Page 13 tab b** | ROI card displays **8.0x**. Card math: 10,752,341 / 1,196,659 = **8.985x**. Off by ~1. | A6 |
| 10 | **Page 13 tab a vs tab c** | Tab a chart x-axis: **Oct 15–16, 2024**. Tab c drift timestamp: **2026-05-10 11:56:50.282**. Same "real-time" page, **~19 months apart**. | A6 |
| 11 | **Page 08** | Drift charts dated May 10 2026 (1.5ms span). Throughput/latency dated Oct 15–16 2024 (24h span). 19-month gap inside one page. | A4 |
| 12 | **Page 05** | Headline "Avg ROI 3.5x". Aggregate 192,155,551 / 50,000,000 = **3.84x**. Two ROIs, one page. | A2 |
| 13 | **Page 05** | KPI "Expected Retained = 118". Same-page Baseline/Current-Selection scenario value = **122**. | A2 |
| 14 | **Page 04** | Apr 2024 cohort: Period 7 = **91.0%** → Period 8 = **92.1%**. Retention is mathematically monotone non-increasing. | A2 |
| 15 | **Page 09** | Two KPI cards both labelled "Avg Expected Uplift": top of page **6.36%** (all 20k), middle of page **10.88%** (treated only). Same label, different scope, no footnote. | A5 |
| 16 | **Page 12** | KPI "Customers Retained" displayed as **`122.29548658078494`** — full Python float spilled to a customer-facing card. | A5 |
| 17 | **Page 10** | "CLV vs Churn Probability" scatter x-axis runs **−0.5 to 1.5**. Probability cannot be negative. | A5 |

---

## 2. Cross-page consistency drift

Same KPI label, different value across pages.

| KPI | P00 | P01 | P05 | P07 | P09 | P10 | P11 | P12 | P14 | P15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Total Customers | 20,000 | 20,000 | — | 20,000 | 20,000 | — | — | — | — | — |
| Avg Churn Prob | 31.31% | 31.31% | — | — | — | — | — | — | — | — |
| Median Churn Prob | — | 15.39% | — | — | — | — | — | — | — | — |
| **High Risk (>50%)** | 5,717 | 5,717 | — | — | — | — | — | — | — | — |
| **Critical (>75%)** | — | 3,596 | — | — | — | — | — | — | — | — |
| **Events Observed (Churn)** | — | — | — | **5,717** ⚠ | — | — | — | — | — | — |
| Total CLV | 57,936,514,970 | — | — | — | — | 57,936,514,970 | — | 57,936,514,970 | — | — |
| Avg CLV | — | — | — | — | — | 2,896,826 | — | 2,896,826 | — | — |
| At-Risk CLV | — | 2,997,471,916 | — | — | — | — | — | 2,997,471,916 | — | — |
| **Overall ROI** | — | — | **3.5x** | — | **9.0x** | — | — | **3.8x** | — | — |
| **Avg Expected Uplift / Avg Uplift** | — | — | — | — | **6.36% / 10.88%** | — | **0.0434** | **0.0434** | — | — |
| Customers Retained / Expected Retained | — | — | **118** (vs scenario 122) | — | — | — | — | **122.29548658…** | — | — |
| Best Model | — | — | — | — | — | — | — | — | ensemble (0.8866) | ensemble (0.8866) |
| **MLflow status** | — | — | — | — | — | — | — | — | **NOT available** ❌ | **Connected: Yes** ✅ |
| **Drift Status** | — | — | — | — | — | — | — | — | — | **RED** (under "All Systems Operational ✅") |

🚨 **Events Observed (Churn) = 5,717 on Page 07 ≡ High Risk count on Page 01** — the survival analysis is using model predictions as observed events. That invalidates the entire KM/hazard output as a real survival signal.

🚨 **Overall ROI** has 3 different values on 3 pages of the same campaign: 3.5x / 9.0x / 3.8x. **Avg Uplift** has 3 different values on 4 pages: 6.36% / 10.88% / 0.0434 / 0.0434. No footnote anywhere reconciles these.

---

## 3. Hidden tiny segments

Page 10 segment table reveals customer counts via Total ÷ Mean:

| Segment | Mean CLV | Total CLV | n (= Total/Mean) |
|---|---:|---:|---:|
| high_value_persuadable | 8,563,357 | 17,126,714 | **2** |
| high_value_lost_cause | 4,153,154 | 4,153,154 | **1** |
| high_value_sure_thing | 8,416,210 | 33,639,590,866 | 3,997 |
| mid_value_sure_thing | 2,443,032 | 16,942,427,813 | 6,936 |
| low_value_sure_thing | 1,171,391 | 2,512,633,819 | 2,145 |
| sleeping_dog | 722,502 | 2,546,096,653 | 3,524 |
| mid_value_persuadable | 1,756,438 | 1,097,773,503 | 625 |
| low_value_persuadable | 424,653 | 1,176,712,448 | 2,771 |

🚨 high_value_persuadable has **n=2** customers but receives an "8.0x ROI" segment label and was the top-ROI segment on Page 12. high_value_lost_cause has **n=1**. The headline ROI distribution is being driven by N=1 and N=2 outliers and the dashboard never surfaces n.

---

## 4. Time-series degeneracy

| Where | x-axis span | Datapoints | Label |
|---|---|---:|---|
| Page 08 Drift Alert Timeline | 11:56:50.282 → .2835 May 10 2026 (~**1.5 ms**) | 1 | "Drift Alert Timeline" |
| Page 08 Mean PSI Over Time | same 1.5ms | 1 | "Mean PSI Over Time" |
| Page 08 Mean KS Statistic Over Time | same 1.5ms | 1 | "Mean KS Statistic Over Time" |
| Page 08 Training Run History | 11:56:50.2892 → .2893 (**0.1 ms**) | 3 | "Model Metrics Across Training Runs" |
| Page 13 tab c PSI/KS/Drift | same 1.5ms | 1 each | "PSI Trend" / "KS Statistic Trend" / "Drift Alerts Over Time" |
| Page 14 Experiment Timeline | **0.1 ms** | 3 | "Model Performance Over Time" |
| Page 14 Learning Rate vs AUC | x: 0.01 / 0.1 / 1 (3 distinct LR), y range 0.0010 | 3 | "degenerate sweep" |
| Page 13 Scoring Volume Over Time | 50 hourly buckets | 50 | **all uniformly = 4** (synthetic placeholder) |

**Rule of thumb:** any chart titled "X Over Time" with a sub-second x-axis is not a trend. 6 charts on the dashboard fail this rule.

---

## 5. Operations / SLO checklist

(From A4 audit)

| Criterion | SaaS Target | Observed | Pass? |
|---|---|---|:--:|
| Avg latency | <50 ms | 19.1 ms | ✅ (number, but data dated Oct 2024) |
| P95 / peak latency | <100 ms | 35 ms peak | ✅ |
| **Error rate** | **<0.1%** | **1.03%** (avg), 1.86% (peak) | ❌ **10× over** |
| Throughput | scalable, real | 49 req/min stub on Oct 2024 timestamps | ❌ |
| **MLflow uptime** | always-on | offline (P14 fallback) AND online (P15 KPI) | ❌ |
| Drift window | ≥7 days | **1.5 ms** | ❌ |
| Error budget tracking | yes | absent | ❌ |
| Probe age per service | yes | absent | ❌ |
| Hyperparameter sweep | real grid | 3 runs at LR=0.1 / training=1.0s | ❌ |
| Model registry stages | Staging/Prod/Archived | absent | ❌ |
| Audit log of model promotions | yes | absent | ❌ |
| Data freshness per page | yes | absent | ❌ |
| Model version on prediction | yes | absent | ❌ |
| Cross-tab consistent clock | yes | 19-month gap | ❌ |

Pass rate: **2 / 14**.

---

## 6. Empty-state vs broken-state confusion

| Page | Defect |
|---|---|
| 06 A/B Testing | 0/0/N/A/0.0% headline KPIs — reads as "broken model". Power calc recommends up to 48,882 participants for MDE 1% with no feasibility guard against the 20k pool. |
| 13 tab a | "Recommended Offer: no_action" banner defaults for every visitor. |
| 13 tab b | Customer lookup returns Priority 1.00 + Uplift 1.46% + no_action — priority field is wired to risk score, not action expected value. |
| 11 | Customer Response Classification pie shows only 2 slices (Persuadable / Lost Cause) but headline KPI uses Persuadable / Sleeping Dogs and table shows 4 segments. Three vocabularies. |

---

## 7. Statistical/data-modeling defects

| Page | Defect |
|---|---|
| 02 | "Best Model: ensemble" declared on **AUC Δ=0.0014** vs DL with no CI / no DeLong test. |
| 02 | All three training_time = **1.0 s** — synthetic floor. AUC vs Training Time chart is structurally degenerate. |
| 02 | Class balance never disclosed; accuracy 0.82–0.87 not benchmarked against trivial-baseline. |
| 01 / 00 | Mean churn 31.31% vs median 15.39% (2.03× ratio), histogram has hard spike at p≈0.9 — bimodal calibration. No reliability diagram, no Brier score. |
| 07 | Median Survival 309 days on a ~350 day horizon = right-censoring artifact. No annotation, no at-risk table, no log-rank, no 95% bands. |
| 07 | 7 of 8 segments show binary Event Rate exactly 0% or 100% — tautological label leak (segment definition encodes outcome). |
| 04 | "Avg Final Retention 2.5%" averages 0.0% trailing cells from cohorts that haven't matured yet. |
| 04 | Only 4 cohorts (Jan–Apr 2024) — below SaaS norm of ≥6. |
| 09 | 16,106 high-priority customers, only 3,398 coupons → **12,708 high-priority customers silently routed to no_action** with no inline reason. |
| 10 | CLV vs Churn x-axis runs **−0.5 to 1.5** (probability can't be negative). |
| 11 | Avg Uplift Score == Avg Treatment Effect (4 decimals) — false equivalence. |

---

## 8. Self-disclaiming banner (P0 framing)

Every page carries:

> 🧪 *"Synthetic data — FULL mode (n=20,000). All KPIs are simulator-generated."*

This is correct disclosure for an internal capstone audit and **wrong framing for a paid SaaS tier**. It tells every prospect "do not trust the numbers on this dashboard". Correct pattern is two-tier UX: **PRODUCTION** mode (real customer data, no banner) vs **DEMO** mode (current banner + watermark + sample-data label).

---

## 9. Top 10 fixes that close the most P0/P1 defects (ranked)

1. **Resolve MLflow status contradiction (P14 vs P15)** — single source of truth for service health, single MLflow client object.
2. **Reconcile Page 02 headline KPI vs confusion matrix** — both should report the same metric on the same test set, with disclosed sample size.
3. **Fix Page 13 Stream depth labels vs lifetime-counter labels** — and align tab clocks.
4. **Single `compute_overall_roi()` helper used by Pages 05 / 09 / 12**, with a cross-page invariant test that fails CI if any two diverge by >1%.
5. **Drop Page 11 duplicate "Treatment Effect" plot** OR compute a genuinely independent ATE from experimental control. Add `assert not np.allclose(uplift, treatment_effect, atol=1e-6)` regression.
6. **Fix Page 11 segment-count math**: 4-segment table sum (19,920) ≠ headline (20,000). Reconcile and normalize the three vocabularies (persuadable/sleeping dog/lost cause/sure thing).
7. **Fix Page 07** — stop using High-Risk count as "Events Observed". Survival should derive from real outcome data.
8. **`format_count(x, currency=...)` helper** + lint banning raw `f"{x}"` for KPI cards. Closes the `122.29548658078494` and `57,936,514,970...` ellipsis cases.
9. **Drift "trend" guard**: render `st.info("Insufficient history (need ≥7 observations)")` when timeseries length <5. Closes Pages 08 / 13c / 14.
10. **Cohort retention monotonicity assertion** on Page 04 + filter trailing 0.0% cells from "Avg Final Retention".

---

## 10. Reference artifacts

### Per-agent audits (full detail)
- `_test_results/iter9_audit_a1.md` — Overview / Churn / Model Performance
- `_test_results/iter9_audit_a2.md` — Segmentation / Cohort / Budget
- `_test_results/iter9_audit_a3.md` — A/B / Survival / Uplift
- `_test_results/iter9_audit_a4.md` — Monitoring / MLflow / System Health
- `_test_results/iter9_audit_a5.md` — Recommendations / CLV / Retention Campaign
- `_test_results/iter9_audit_a6.md` — Real-Time Scoring (3 tabs)

### Per-page structured data dumps (this iteration)
- `_test_results/page_data/00_overview.md` … `15_system_health.md` (16 files; one combined `13_realtime_scoring.md` covers the 3 tabs)

### Source PNGs (all 18 freshly captured this iteration with full content)
- `_test_results/dashboard_pages/00_overview.png` … `15_system_health.png` (15 pages + 3 tabs of page 13 = 18 PNGs)

---

**Bottom line:** 6 independent reviewers (no shared context) landed on 6 NOT-SHIP verdicts. Cataloged: **17 same-page or cross-page contradictions**, **3 different ROI definitions for the same campaign**, **3 different "Avg Uplift" values for the same population**, **6 sub-second "trend over time" charts**, **2 hidden segments with n≤2 driving headline ROIs**, and a **survival analysis using predictions as observed events**. Internal capstone-demo with verbal disclaimer = OK; paid SaaS pilot today = procurement death in the first 30 minutes.

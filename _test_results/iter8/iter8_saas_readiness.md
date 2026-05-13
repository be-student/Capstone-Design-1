# Iter 8 — Live Dashboard SaaS-Readiness Audit (zero-context, 6 agents)

**Date:** 2026-05-10
**Method:** Captured 18 full-page Playwright screenshots from the **live** dashboard at `http://localhost:8501` (FULL mode, n=20,000), replaced the existing PNGs in `_test_results/dashboard_pages/`, then dispatched 6 sub-agents in parallel — **each with no prior context on this project** — and asked each to look at 3 PNGs and judge as a SaaS-buyer would.
**Aggregate verdict:** **6/6 agents DO-NOT-SHIP** for paid tier.

---

## 0. Coverage map

| Agent | Pages |
|---|---|
| A1 | 00 Overview · 01 Churn Analytics · 02 Model Performance |
| A2 | 03 Customer Segmentation · 04 Cohort Analysis · 05 Budget Optimization |
| A3 | 06 A/B Testing · 07 Survival Analysis · 11 Uplift Modeling |
| A4 | 08 Model Monitoring · 14 MLflow Experiments · 15 System Health |
| A5 | 09 Recommendations · 10 CLV Prediction · 12 CLV & Retention Campaign |
| A6 | 13 Real-Time Scoring (3 tabs: Live / Offers / Monitoring) |

(15 distinct dashboard pages + page 13's three tabs = 18 PNGs, all freshly captured this session.)

---

## 1. Verdict by domain

| Agent | Domain | Verdict |
|---|---|:--:|
| A1 | Overview / Churn / Model Perf | **DO-NOT-SHIP** |
| A2 | Segmentation / Cohort / Budget | **DO-NOT-SHIP** |
| A3 | Experimentation / Survival / Uplift | **DO-NOT-SHIP** |
| A4 | Operations (Monitoring / MLflow / Health) | **DO-NOT-SHIP** |
| A5 | Money (Reco / CLV / Campaign ROI) | **DO-NOT-SHIP** |
| A6 | Real-Time Serving | **DO-NOT-SHIP** |

---

## 2. Top blockers — wrong / unreliable / missing

Each row is something an enterprise procurement reviewer can defeat the dashboard with on first look. Severity is the orchestrator's call across the 6 agents' independent findings.

### P0 — visible math / contradiction errors (instantly defeats trust)
| # | Where | Defect | Why it's a P0 |
|---|---|---|---|
| 1 | Page 14 vs Page 15 | Page 14 banner: *"MLflow tracking server not available — showing cached experiment data."* Page 15 KPI: *"MLflow Tracking — Connected: Yes."* | Two pages of the **same product** disagree about whether a service is up. Not a UX issue — a literal contradiction. |
| 2 | Page 13 tab a | "Request Stream: 0" + "Response Stream: 0" KPI cards, with the chart below them showing 20–80 req/min. | Either the streams are dead and the chart is a lie, or the chart is real and the KPI cards are dead. Either way, one of them is broken. |
| 3 | Page 13 tab b | Card says "ROI 8.0x". Card math: 10,752,341 / 1,196,659 = **8.985x**. | Off by a full hundredth — round-up bug or stale cache. A CFO will catch this in 15 seconds. |
| 4 | Page 05 | At least 2 segment rows show **Expected Retained = 0 with Revenue Saved > 0**. | Mathematically impossible if "Revenue Saved" is "CLV × P(retain | treated)". Shows the formula doesn't match the table. |
| 5 | Page 05 | "Avg ROI 3.5x" vs aggregate 192.16M / 50M = **3.84x**. | Two ROI definitions inside one page (mean-of-segments vs aggregate) — neither is footnoted. |
| 6 | Page 09 vs Page 12 | "Avg Expected Uplift" = **6.36% on P09** vs "Avg Uplift" = **4.34% on P12** for the same n=20,000. | Same population, two pages, no reconciliation. Definition trap. |
| 7 | Page 04 | Cohort Jan-2024 retention: Period 9 = 86.5% → Period 10 = **88.3%**. | Retention is mathematically monotone non-increasing. An 88.3% follow-up after 86.5% is impossible — almost certainly a denominator-shrink artifact being averaged in without filter. |
| 8 | Page 13 tab a vs tab c | Tab a chart x-axis shows **Oct 15–16, 2024**; tab c drift timestamp is **2026-05-10 11:56:50**. | Same "real-time" page, ~19 months apart. Operators cannot tie offers ↔ scores ↔ drift to one consistent artifact. |
| 9 | Page 00 vs Page 01 | Same 20,000-customer roster, but Overview's leftmost-bin probability count ≈ 4,000 while Churn Analytics's ≈ 3,500. | Two pages render the same histogram with different counts. Cache divergence or different filters silently applied. |
| 10 | Page 11 | Avg Uplift Score **0.0434** == Avg Treatment Effect **0.0434**, with visually identical histograms below. | One variable plotted as two — the dashboard's "treatment_effect" is just an alias for "uplift_score", but it's presented as if it's an independent estimate. |

### P1 — degenerate / single-point "trend" charts
| # | Where | Defect |
|---|---|---|
| 11 | Page 08 Drift Alert Timeline | **1 red dot** at y=7. Total Checks = 1, Status = RED. X-axis spans `11:56:50.282 → .2835` — that's **1.5 milliseconds**. |
| 12 | Page 14 Hyperparameter scatter | 3 MLflow runs, all at LR=0.1000, training_time=1.0s, AUC spread Δ=0.0014. Plotted as a "Learning Rate vs AUC" scatter with all 3 points stacked on one X. |
| 13 | Page 14 Model Performance Over Time | 3 timestamps **40 microseconds apart** plotted as a temporal series. Three sequential function calls dressed as longitudinal data. |
| 14 | Page 13 tab c | Single drift datapoint plotted on a 1.5-millisecond x-axis. Labelled "Latest Alert Level: RED" with 1/1 red rate. Textbook "single-observation extrapolated as trend" failure. |
| 15 | Page 04 | Only **4 cohorts** (Jan–Apr 2024). Industry SaaS norm is ≥6 monthly cohorts for any retention narrative; 4 makes the heatmap, period-over-period delta, and per-cohort lines structurally underpowered. |

### P2 — empty states presented as broken state
| # | Where | Defect |
|---|---|---|
| 16 | Page 06 A/B Testing | Total Experiments = 0, Significant Results = 0, Avg Lift = 0.0%, Best Experiment = N/A. Power calc says 1,812 needed, never reconciled against the 20k pool. Reads as "the model is broken", not "no experiments yet". |
| 17 | Page 13 tab b | 44 offers shown without a **denominator** (44 of which 20,000? per minute? lifetime?). |
| 18 | Page 11 | 4-quadrant uplift segmentation collapsed to 2: Persuadable 16,317 + Sleeping Dogs 3,683 = exactly 20,000. Sure-Thing and Lost-Cause are missing entirely. No coupon-eligibility guardrail visible to keep Sleeping Dogs (negative uplift) out of treatment. |

### P3 — format / precision / labelling leaks
| # | Where | Defect |
|---|---|---|
| 19 | Page 10 / 12 | "Total CLV: 57,936,514,970..." with literal trailing **ellipsis** — the format helper truncates instead of using thousand-separator units (₩57.9B). At-Risk CLV "2,997,471,916 K…" same bug. No currency label visible above the fold. |
| 20 | Page 10 vs histogram annotation | Median CLV KPI tile = **1,701,727**; histogram annotation = **1,701,826**. ~99 KRW drift between two views of the same statistic — sampling/rounding inconsistency leaks to UI. |
| 21 | Page 13 tab a | Latency right-axis range **0…3.5** with no unit. Could be ms, s, or %. A 3.5% error rate would be invisible. |
| 22 | Page 02 | "Best Model: ensemble" declared on a **0.0006 AUC margin**. No CI, no significance test, no operating-point threshold, no calibration. ML wins recall, DL wins precision/F1/accuracy — the choice of "best" is arbitrary. |
| 23 | Page 03 | Headline KPI "Highest Risk Segment = dormant" with **no risk score shown** next to the segment name; segment definitions and per-segment churn probabilities are missing from the page. |

### P4 — production-grade things that are simply not there
| # | What's missing | Pages where it should appear |
|---|---|---|
| 24 | Data freshness timestamp ("data as of …") | every page |
| 25 | Model version / model lineage | 02, 08, 09, 13a, 14 |
| 26 | Confidence intervals | 01 (probabilities), 02 (AUC), 04 (retention), 07 (KM curves), 11 (uplift) |
| 27 | Class-balance disclosure | 02 (accuracy 0.8993 across 3 models is dominated by the imbalance) |
| 28 | At-risk table for KM curves | 07 |
| 29 | Right-censoring annotation on median duration | 07 (309 days near 365-day horizon = censoring artifact, not a true median) |
| 30 | SLO panels (error rate, p50/p95 latency, throughput, probe age) | 08, 13c, 15 — all six SLO checklist items are absent |
| 31 | Cohort denominator (#customers per cohort × period) | 04 |
| 32 | Channel breakdown | 05 (page header implies one, body has nothing) |
| 33 | Overall ROI / Revenue Saved / Customers Retained tiles above the fold | 12 — the page is literally called "CLV & Retention Campaign" and these tiles are absent |
| 34 | Quick-lookup / per-customer prediction explainer | 13 — promised by tab title, not visible |
| 35 | Filter pills echo into KPI cards | 13b — filters change segment without re-aggregating the headline 44 / cost / ROI cards |

### P5 — self-disclaimed synthetic data
| # | Defect |
|---|---|
| 36 | **Every page** carries a banner: *"Synthetic data — FULL mode (n=20,000). All KPIs are simulator-generated."* The banner is correct disclosure for an internal audit and **wrong framing for a paid SaaS tier** — it tells every prospect "do not trust the numbers on this dashboard". The right pattern is a two-tier UX: real customer data path with no banner, simulator/demo path with a clearly-toggled "DEMO" watermark. |

---

## 3. Per-domain summary (one paragraph each)

**A1 — Overview / Churn / Model Performance.** AUC margin between the three models is 0.0006; "Best Model: ensemble" is declared without a significance test. Risk-tier counts contradict each other: Critical(>75%) = 3,596 is 63% of High(>50%) = 5,717, which means most "High-risk" customers are actually "Critical" — the threshold scheme is mis-stratified. Mean churn probability 31.31% is ~2× the median 15.39% (skewed distribution mis-represented as average). The histogram's leftmost bin renders ~4,000 on Overview but ~3,500 on Churn Analytics for the same 20k roster. Total CLV truncates with ellipsis and has no currency.

**A2 — Segmentation / Cohort / Budget.** Cohort retention violates monotonicity (86.5% → 88.3%, period 9→10) — denominator shrinks past the period horizon and the dashboard averages 0.0% trailing cells into "Avg Final Retention 2.5%". Only 4 cohorts (Jan–Apr 2024) plotted. Budget LP allocates 31,000 KRW to a segment with claimed 8.00x ROI while pushing 70% of the 50M budget to lower-ROI mid/low segments — either the LP solver is broken or the displayed ROIs aren't what the LP optimised against. "Avg ROI 3.5x" disagrees with aggregate 192.16M/50M = 3.84x. Two segment rows show "Expected Retained = 0" alongside positive Revenue Saved.

**A3 — A/B / Survival / Uplift.** A/B page on a 20k pool shows 0/0/0.0% as headline KPIs with no empty-state framing — reads as broken model. Power-analysis card says 1,812 needed but never reconciles against the 20k available. Survival median 309 days at 28.59% event rate near a 365-day horizon = right-censoring artifact, no annotation, no CIs, no at-risk table, no log-rank. Uplift score 0.0434 == treatment effect 0.0434 with visually identical histograms — false equivalence (one variable, two charts). 4-quadrant uplift segmentation collapses to 2 categories (Persuadable + Sleeping Dogs sum to exactly 20,000). No Sleeping-Dog (n=3,683 negative uplift) coupon-eligibility guardrail.

**A4 — Operations.** Drift timeline is a single red dot at y=7 on a 1.5-millisecond x-axis. MLflow runs are 40μs apart with all 3 sharing LR=0.1 and training_time=1.0s — degenerate sweep, AUC Δ=0.0014. Page 14 banner says "tracking server not available, showing cached"; Page 15 KPI says "MLflow: Connected, Yes". **The dashboard is internally inconsistent about whether MLflow is up.** No SLO panels anywhere — error rate, latency, throughput, drift window, probe age all absent.

**A5 — Money.** Total CLV "57,936,514,970..." truncates with literal ellipsis on Pages 10 and 12. At-Risk CLV "2,997,471,916 K..." same bug. Avg Expected Uplift = 6.36% on Page 09 vs Avg Uplift = 4.34% on Page 12 for the same population — no footnote, no reconciling formula. Three different ~16k cohort sizes are not reconciled: High Priority 16,106 / Treatable 16,317 / no_action 16,602. 16,106 high-priority customers but only 3,398 receive a coupon → 12,708 "high priority + no_action" with no explanation. Page 12 ("CLV & Retention Campaign") has **no Overall ROI, no Total Revenue Saved, no Customers Retained tile above the fold** — the centerpiece KPIs are missing from the centerpiece page.

**A6 — Real-Time Serving (3 tabs).** Banner on every tab: "simulator-generated", so nothing on this page is "real-time". Tab a's chart x-axis shows Oct 15–16 2024 throughput; tab c's drift datapoint is timestamped May 10 2026 — same product, **~19 months apart**. Tab a Request/Response Stream KPI cards show 0/0 directly below a chart showing 20–80 req/min — direct contradiction. Tab b card says ROI 8.0x but card math = 8.985x. Tab b shows 44 offers without denominator. Tab c plots one observation on a 1.5ms x-axis labelled "Latest RED, 1/1 red-alert rate" — textbook fail. No model_version on any of the three tabs; no PSI/KS values surfaced; the only KPI that recurs across tabs is the synthetic-data banner.

---

## 4. Cross-page consistency contradictions found this iteration

| KPI | Page A | Page B | Mismatch |
|---|---|---|---|
| Avg Uplift | 6.36% (P09 "Avg Expected Uplift") | 4.34% (P12 "Avg Uplift") | same n=20k, two values |
| MLflow service status | "not available" (P14 banner) | "Connected: Yes" (P15 KPI) | direct contradiction |
| Median CLV | 1,701,727 KRW (P10 KPI tile) | 1,701,826 KRW (P10 histogram annotation) | ~99 KRW drift on the same page |
| Page 13 time anchor | Oct 15–16, 2024 (tab a chart) | May 10, 2026 11:56:50 (tab c) | 19-month gap |
| ROI denominator | 3.5x (P05 mean-of-segments) | 3.84x (P05 aggregate) | two values one page, no footnote |
| ROI math | 8.0x (P13b card) | 8.985x (P13b card-math) | rounding/stale display |
| Histogram bin counts | ~4,000 leftmost (P00) | ~3,500 leftmost (P01) | same 20k roster |
| Risk-tier overlap | Critical(>75%) = 3,596 (P00) | High(>50%) = 5,717 (P00) | 63% of "high" is actually "critical" — strata not mutually-exclusive in a way the chart implies |
| Total Population in cohort sizes | 16,106 / 16,317 / 16,602 / 20,000 | (P09, P11, P12) | 4 different population sizes, no reconciling crosswalk |

---

## 5. What an external SaaS pilot would need before "GO"

Concrete remediation order. Each item closes one or more of the P0/P1 findings above.

1. **Resolve the MLflow online/offline contradiction** between Page 14 and Page 15 — single source of truth for service health.
2. **Fix the Page 13 contradictions** — Stream depth labels separated from lifetime-counters, time anchors aligned across tabs, ROI math display = ROI math computation.
3. **Fix the cohort retention monotonicity violation** on Page 04 — filter trailing 0.0% cells out of "Avg Final Retention", and produce ≥6 monthly cohorts before shipping the page.
4. **Page 05 LP** — debug why an 8x-ROI segment receives 31k KRW; either the displayed ROI is post-allocation expected-marginal (in which case label it that way) or the LP isn't reading what's on screen.
5. **Page 11 uplift double-plot** — drop the duplicate Treatment-Effect KPI/scatter or compute a genuinely independent ATE from the experimental control. Add a 4-quadrant guardrail so Sure-Thing and Lost-Cause segments populate.
6. **`format_currency(n)` helper** — bans truncation+ellipsis, thousand-separator everywhere, currency symbol in every CLV/Revenue tile.
7. **Single `compute_overall_roi(...)` helper** with cross-page invariant test. Same value on Page 05 / 09 / 12.
8. **Empty-state component** — Page 06 zero-experiments, Page 13b filter-not-applied, etc. should render `st.info("No data yet — run X to populate")` instead of zero KPIs.
9. **Drift `n < 5` guard** — replace single-point "trend" charts with `st.info("Insufficient history — need ≥7 observations")` on Pages 08 / 13c.
10. **MLflow real hyperparameter sweep** — 3 runs at LR=0.1 / epochs=1 isn't a sweep. Either run a real grid or remove the "Hyperparameter Analysis" section.
11. **Add data-freshness timestamp, model version, and confidence intervals** to every page that asserts a number (33 of the 35 visible KPIs lack at least one of these).
12. **Two-tier UX**: "DEMO" mode (current banner) vs "PRODUCTION" mode (no banner, real customer data). The current always-on banner self-disqualifies the dashboard for a paid pilot.
13. **A/B page**: surface lift-realism guardrail (warn if observed lift outside 5–15% literature band) plus an empty-state callout when there are no experiments.
14. **Survival page**: add right-censoring annotation, at-risk table under the KM curves, and 95% CIs.

---

## 6. Reference artifacts

### Per-agent detailed audits
- `_test_results/iter8_audit_a1.md` — Overview / Churn / Model Performance
- `_test_results/iter8_audit_a2.md` — Segmentation / Cohort / Budget
- `_test_results/iter8_audit_a3.md` — A/B / Survival / Uplift
- `_test_results/iter8_audit_a4.md` — Monitoring / MLflow / System Health
- `_test_results/iter8_audit_a5.md` — Recommendations / CLV / Retention Campaign
- `_test_results/iter8_audit_a6.md` — Real-Time Scoring (3 tabs)

### Source PNGs (replaced this session via Playwright fullPage capture)
- `_test_results/dashboard_pages/00_overview.png` (NEW)
- `_test_results/dashboard_pages/01_churn_analytics.png`
- `_test_results/dashboard_pages/02_model_performance.png`
- `_test_results/dashboard_pages/03_customer_segmentation.png`
- `_test_results/dashboard_pages/04_cohort_analysis.png`
- `_test_results/dashboard_pages/05_budget_optimization.png`
- `_test_results/dashboard_pages/06_ab_testing.png`
- `_test_results/dashboard_pages/07_survival_analysis.png`
- `_test_results/dashboard_pages/08_model_monitoring.png`
- `_test_results/dashboard_pages/09_recommendations.png`
- `_test_results/dashboard_pages/10_clv_prediction.png`
- `_test_results/dashboard_pages/11_uplift_modeling.png`
- `_test_results/dashboard_pages/12_clv_retention_campaign.png`
- `_test_results/dashboard_pages/13_realtime_scoring_a_live.png`
- `_test_results/dashboard_pages/13_realtime_scoring_b_offers.png`
- `_test_results/dashboard_pages/13_realtime_scoring_c_monitoring.png`
- `_test_results/dashboard_pages/14_mlflow_experiments.png`
- `_test_results/dashboard_pages/15_system_health.png` (NEW)

---

**Bottom line:** every domain came back DO-NOT-SHIP **independently**, with **no shared context between agents**. Six different reviewers, six different DO-NOT-SHIPs, ten direct cross-page contradictions catalogued. A paid SaaS pilot today would die in 30 minutes of executive walkthrough. Internal capstone-demo OK with a verbal disclaimer; external customer demo not OK.

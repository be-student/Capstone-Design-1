# Pass 1 — Realism Audit Report

> Methodology note up front: `mcp__playwright__browser_navigate` was denied at the session level, as were Bash, PowerShell, and Write. The Streamlit dashboard could not be driven directly. Audit was performed by reading the exact JSON/CSV artifacts the dashboard renders (`data/artifacts/*`, `data/raw/*`) — i.e. the source-of-truth that `src/dashboard/data_loader.py` feeds into every chart.

## Pages visited (from page registry in `src/dashboard/app.py:4478-4495`)
Overview, Churn Analytics, Model Performance, Customer Segmentation, Cohort Analysis, Budget Optimization, A/B Testing, Survival Analysis, Model Monitoring, Recommendations, CLV Prediction, Uplift Modeling, CLV & Retention Campaign, Real-Time Scoring, MLflow Experiments, System Health.

## Page-by-page findings

### Model Performance / Overview (`model_metrics.json`)
- ML AUC=**1.000**, Acc/Prec/Rec/F1=**1.000** — **Implausible**. Real churn AUC is 0.70–0.88. Textbook target leakage.
- Ensemble AUC=**1.000** — **Implausible**, same reason.
- DL Transformer val AUC=**0.9993** at epoch 4, monotone train loss, no realistic noise — **Implausible**. Production transformers plateau at 0.85-0.92 with jitter.
- DL test AUC=0.9986, F1=0.967 — **Implausible**.

### Churn Analytics / Real-Time Scoring (`churn_predictions.csv`)
- Probabilities are **bimodal**: every row is ~1.04e-5 OR ~0.99996, no middle ground. **Implausible** as a probabilistic classifier — it is a class indicator. Breaks calibration, threshold tuning, expected-value math.

### Feature Importance (`feature_importance.csv`)
- `recency` gain=**6.736** vs next feature 0.361 → recency carries **~88%** of total importance. **Implausible-leakage**: the churn label is *defined* as "no purchase for N days", which is exactly recency. The model is memorizing the label definition.

### Survival Analysis (`survival_results.json`)
- C-index=**0.9592** — **Implausible**. Realistic customer-survival C-index is 0.65–0.78.
- median_survival_prob_90d=**5.35e-36** — **Implausible**, contradicts the 22.8% observed churn rate. Curve is broken.
- 1,140 events / 5,000 customers — small but acceptable.

### CLV Prediction (`clv_validation.json`, `clv_distribution.json`)
- Holdout R²=**0.7832** — **Suspicious**. Real e-commerce CLV R² is 0.3–0.6. Annualization factor is **8.30×** (44-day window scaled to 12 months) which mechanically inflates the score.
- MAE 1.25M / RMSE 1.93M KRW — ratio reasonable. Distribution shape (mean 3.35M, p95 13.83M, max 17.84M) is **Realistic**.

### Customer Segmentation (`segment_summary.csv`, `segment_validation.json`)
- 6 segments with counts {**4**, 996, 1064, 936, 72, 1928} — **Suspicious**. `high_value_persuadable` = **n=4**. Action plan resting on 4 customers is not deployable.
- Intra-segment avg churn is either ~0.000 or ~0.999 — **Implausible** homogeneity, confirms bimodal output.

### Uplift Modeling (`uplift_results.csv`)
- Uplift scores collapse to ~**5 discrete values** (0.0028, 0.0418, 0.0449, 0.0595, 0.1835) across 5,000 rows — **Implausible**, looks like a persona→treatment-effect lookup table.
- `positive_uplift_count` = **5,000/5,000 (100%)** — **Implausible**, real campaigns always have negative-uplift customers.

### A/B Testing (`ab_test_detailed.json`)
- 2 experiments, lift **33.8%** and **30.1%** churn reduction, p=5.3e-15 and 2.4e-12, power≈1.0 — **Suspicious**. Literature lift typically 5–15%. Not as damning as 20+ tests at p<0.001 but still rosy.

### Cohort Analysis (`cohort_retention_matrix.csv`, `cohort_milestones.csv`)
- Only **2 cohorts**. 2024-02 drops from 0.901 → **0.000** at month 5 — observation-window cliff, **Implausible** as retention.
- M6/M12 milestones are **carry-forward fallbacks** of M4/M5 (`cohort_analysis.json` admits this). Any "12-month retention" KPI is an extrapolation, not a measurement.

### Budget Optimization (`budget_optimization_summary.json`)
- ROI 50%/100%/200% budget = 1.73 / 0.87 / 0.43 — diminishing-returns shape is **Realistic**.
- Retained value barely moves 100% → 200% (43.49M → 43.49M) — **Suspicious**, retention saturation flowing from the bimodal-probability artifact.
- `high_value_persuadable` ROI=8.51 on **n=4** — too small to act on.

### Model Monitoring (`monitoring_report.json`)
- All features green, max PSI ≈ 0.016 vs 0.10 yellow — **Realistic** for same-distribution holdout.

### System / Required Artifacts (`required_artifacts_checklist.json`, `generation_summary.json`)
- `full_submission_ready=false`, `group_size_passed=false`, mode=**small (5k/20k target)**. The system itself flags it is sub-spec, but the dashboard renders KPIs as if they were full-scale.

## Top 5 most suspicious findings

1. **Churn AUC=1.000 (ML & Ensemble), DL≈0.999** plus `recency` carrying 88% of feature importance — unambiguous target leakage (label is defined as recency-threshold).
2. **Bimodal churn probabilities (~1e-5 vs ~0.9999, nothing between)** — a class indicator dressed as a probability; breaks every downstream pipeline that expects a calibrated score.
3. **Survival C-index=0.959 with median 90-day survival prob=5e-36** — two physically inconsistent numbers; survival page is not deployable.
4. **Uplift collapses to 5 discrete values, 100% positive uplift, "high_value_persuadable" segment = n=4** — primary actionable list is 4 customers wide.
5. **Cohort M6/M12 = carry-forward of M4/M5, only 2 cohorts, 2024-02 cliff to 0%** — any 12-month retention KPI shown without that disclaimer is misleading.

Honorable mentions: A/B lift 30–34% (high end), CLV R²=0.78 (annualization-inflated), pipeline self-flag `full_submission_ready=false` not surfaced.

## Overall SaaS-readiness verdict

**DO-NOT-SHIP** as-is. A CTO seeing AUC 1.000, C-index 0.96, R² 0.78 and 100% positive-uplift on 5k synthetic customers would either lose trust in the team or — worse — believe it. The numbers individually trip every textbook leakage tripwire and collectively misrepresent real-world performance. Minimum gates: (1) persistent "Synthetic data, SMALL mode (5k/20k); not representative of production performance" banner on every page; (2) fix the recency-driven label leak so probabilities are no longer bimodal. With those two fixes it could ship as **NEEDS-DISCLAIMER** for a sandbox/pilot, never as a production churn product.

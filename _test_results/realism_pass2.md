# Pass 2 — Independent Realism Audit

## Methodology
- Tools available: Read, Grep, Glob. Bash, `mcp__ide__executeCode`, and Write were denied; Playwright was loaded but unnecessary because file reads cover the artifact set. Cross-artifact arithmetic done with Grep `count` mode against CSVs.
- Artifacts inspected: `data/raw/generation_summary.json`, `data/artifacts/{model_metrics,threshold_analysis,feature_importance.csv,churn_predictions.csv}`, all `results/*.json` plus the listed CSVs (`segment_summary`, `cohort_retention_matrix`, `uplift_results`, `recommendations`, `clv_actual_vs_predicted`, `clv_distribution`), and `src/dashboard/app.py`.

## Cross-artifact consistency findings
- **Empirical churn vs generator label** — `churn_predictions.csv` flags 1,138 customers as `critical` (prob >0.99) out of 5,000 = 0.2276, matching `generation_summary.churn_rate = 0.228` to three decimals. Internally consistent — but suspiciously perfect (see leakage below).
- **Survival vs cohort** — `survival_results.median_survival_prob_90d = 5.3e-36` (effectively 0). `cohort_retention_matrix.csv` at M3 shows 0.95 / 0.95 retention (5% churn at 90 days). **Direct contradiction by ~30 orders of magnitude.** The dashboard surfaces only the survival number, hiding the conflict.
- **A/B vs uplift mean** — `ab_test_results.lift = 0.338` (relative, 27.4% → 18.2%). Mean per-row `uplift_score` is ~0.05–0.07 absolute. Aggregate A/B lift is plausible but is not derivable from the per-customer uplift table.
- **Segment summary churn rates** — match `churn_predictions.csv` consistently, but only because the underlying probabilities are bimodal (averages are essentially 0 or 1, meaningless).
- **CLV high-value threshold (3.78M KRW)** matches `clv_distribution.p80 = 3.778M` exactly — internally consistent.

## Missing / hidden disclosures
- **No "synthetic / SMALL mode" banner.** Grep on `src/dashboard/app.py` and `src/dashboard/` returns zero hits for `synthetic`, `small_mode`, `simulator`, `simulated`, `disclaimer`, or `data_source`. A customer would never know this is simulator output.
- **No calibration plot.** A binary classifier with AUC = 1.0 and no reliability diagram is a red flag.
- **No CIs on KPI cards.** `confidence_interval` is referenced once (app.py:1206, A/B test) but churn KPIs, CLV MAE/RMSE/R², survival median, and uplift segment counts are point estimates only.
- **Decision threshold partially tunable.** A 0.78 AUC gate is hard-coded at app.py:356; `classify_risk` cutoffs are not exposed. Only the analytics-view churn-probability slider (app.py:3241) is user-adjustable.
- **Holdout disclosure.** `clv_validation.holdout_size = 1000` exists in JSON but isn't surfaced. `churn_predictions.csv` shows `split = train` for all 5,000 rows — the churn metrics likely reflect train-set scoring, not held-out generalization.
- **`generation_summary.validation.group_size_check.passed = false`** with `Small mode summary only` warning — failure is not surfaced anywhere on the dashboard.

## Distributional sanity
- **Churn probability is brutally bimodal.** Of 5,000 rows: 3,164 have prob `1.x e-05` (~0), 1,108 are 0.999..., and only **9 of 5,000** are classified `medium`/`high`. There is no middle. Textbook uncalibrated/leakage signature.
- **`feature_importance.csv` is the smoking gun.** `recency` carries 6.74 of total ~7.68 importance (87.7%); `purchase_cycle_anomaly`, `frequency`, `days_since_last_purchase` together cover the rest. Since the churn label is *defined* by recency, the model is essentially predicting `y` from `y` — explaining AUC = accuracy = F1 = 1.0 on the ML head and DL AUC = 0.999.
- **Uplift scores are a tiny lookup table.** Three values 0.04179, 0.04492, 0.18348 alone account for 1,246 + 782 + 730 = **2,758 / 5,000 (55%)** of all customers. The first 25 rows show only 5–6 distinct scores. Top-K targeting is degenerate — every cutoff promotes/demotes huge equivalence classes at once.
- **Cohort retention is monotone non-increasing per cohort.** Cohort 2024-02 dropping from 0.90 (M4) to 0.00 (M5) is a censoring/fallback artifact (per the carry-forward policy note in `cohort_analysis.json`), not a real cliff — but a non-technical viewer would not know this.

## Operational realism
- **High-value-actionable segment is unactionable.** `segment_summary.csv` shows only **4 customers** in `high_value_persuadable`, 72 in `mid_value_persuadable`. Anything <50 cannot be safely A/B-tested or staffed for outreach — and the marquee actionable segment is 4 names.
- **Budget optimizer is saturated.** 50% budget retains 43.21M KRW; 100% retains 43.49M; 200% retains 43.49M — doubling spend recovers <0.01% more. ROI 1.73 / 0.87 / 0.43 across the three scenarios. The headline 100% ROI of 0.87 is the worst of the three; the optimizer is correct that 50% dominates, but the dashboard does not lead with that insight.
- **Recommendations push coupons to zero-risk customers.** `recommendations.csv` issues `coupon` to all inspected rows including `high_value_sure_thing` customers at 0% churn risk (e.g. C000006 at p_churn = 1e-5). Pure discount cannibalization — the recommender does not gate on `churn_probability`.

## Top 5 most suspicious findings (rank-ordered by severity)
1. **Label leakage in churn model** — `recency` = 87.7% of importance; AUC/F1 = 1.0; predictions are 0/1 bimodal. The model is a tautology with respect to the churn label and will collapse on real data.
2. **Survival vs cohort retention contradiction** — `median_survival_prob_90d ≈ 5e-36` vs cohort M3 retention ≈ 0.95. Two artifacts disagree by ~30 orders of magnitude; the dashboard suppresses the conflict.
3. **Uplift "model" is a 6-value lookup table** — top three uplift values cover 55% of the customer base; rank-based targeting cannot work.
4. **No data-source banner** — dashboard nowhere discloses synthetic / SMALL mode origin, the group-size validation failure, or that all rows are `split = train`.
5. **Recommender issues coupons to zero-risk customers** — wastes retention budget on customers who weren't leaving.

## Overall SaaS-readiness verdict
**DO-NOT-SHIP.** Beneath polished numbers (AUC 1.0, p = 5e-15, ROI 0.87) sits a churn model whose feature importance shows it memorized the label, an uplift model with a ~6-value codomain that cannot drive ranked targeting, a survival/cohort contradiction of 30 orders of magnitude, a "high-value-actionable" segment of literally 4 customers, and recommendations that issue coupons to zero-risk customers. There is no synthetic-data banner, no calibration plot, no KPI confidence intervals, and the only `split` value present is `train`. A CTO would not — and should not — sign off: the system would be debunked by any data-literate buyer or, worse, used to spend real retention budget on top of leaky, unranked predictions. Minimum fixes: retrain without the label-defining feature, replace the lookup-table uplift with a real meta-learner, reconcile survival with cohort, add a "synthetic — not for production decisions" banner, and surface train/holdout split + calibration plot on the metrics page.

## Disagreements with what a typical first-pass auditor might say
- **AUC = 1.0 isn't automatically fraud — in this synthetic dataset it is the expected ceiling** because the simulator deterministically generates churn from persona + recency. So I would not flag perfect AUC alone as the smoking gun; it's a leakage signal that *will* disappear on real data, but the damning evidence is `feature_importance.csv` + the bimodal output, not the AUC itself.
- **Budget ROI of 0.87 looks weak but is mathematically sane** — saturation at 100% budget is real (per the what-if scenarios). The fix is showing 50%-budget ROI of 1.73 as the lead metric, not "optimizer is broken."
- **The 0.0 retention in cohort 2024-02 M5 is a censoring/fallback artifact**, disclosed (poorly) in `cohort_analysis.json::milestone_fallback_policy`. A first-pass auditor might flag this as a churn cliff; it isn't.
- **Survival concordance 0.959 is high but defensible** on synthetic data with strong recency signal — what's not defensible is the 5e-36 median 90-day survival, which contradicts the rest of the artifact set, not the C-index.

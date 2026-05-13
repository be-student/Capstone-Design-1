# Realism Synthesis — Final Verdict (Pass 1 + Pass 2)

**Date:** 2026-05-08
**Method:** Two independent fresh-context auditors (Pass 1 = chart-by-chart readthrough, Pass 2 = cross-artifact consistency + missing-disclosure analysis). Playwright was denied to both subagents; both fell back to reading the same JSON/CSV artifacts that the dashboard renders, so the audit applies to whatever chart consumes those files.

## 1. Convergent Verdict

**Both auditors independently returned `DO-NOT-SHIP`.** Two reviewers, separate angles, same conclusion — that's the strongest possible signal short of a full third pass.

| Question | Pass 1 | Pass 2 |
|---|---|---|
| SaaS-ready? | DO-NOT-SHIP | DO-NOT-SHIP |
| Most suspicious item | AUC = 1.000 + recency = 88% importance | Same — label leakage |
| Bimodal probabilities flagged? | Yes (3,164 ~0, 1,108 ~1) | Yes (only 9 / 5,000 in middle band) |
| Uplift score collapse flagged? | Yes (5 discrete values) | Yes (top 3 cover 55% of customers) |
| n=4 high_value_persuadable flagged? | Yes | Yes |
| Cohort M6/M12 carry-forward flagged? | Yes | Yes (defended as known censoring) |

## 2. Convergent Findings (Both Passes Agreed — Highest Confidence)

| # | Finding | Severity |
|---|---|---|
| 1 | **Label leakage**: `recency` = 87.7% of feature importance, churn label is *defined* by recency. Model memorizes the label. | **Critical** |
| 2 | **Bimodal churn probabilities** (~1e-5 vs ~0.9999, almost nothing in the 0.05–0.95 band). It's a class indicator, not a calibrated probability. | **Critical** |
| 3 | **AUC = 1.000 / Acc = 1.000 / F1 = 1.000** for ML & Ensemble, DL = 0.999. | Critical (downstream of #1) |
| 4 | **Uplift score collapse to ~5–6 values** — degenerate top-K targeting. | **Critical** |
| 5 | **`high_value_persuadable` segment = n=4 customers** — primary actionable list is too small to operationalize. | High |
| 6 | **Cohort M6/M12 are carry-forward fallbacks** (admitted in `cohort_analysis.json::milestone_fallback_policy`). 12-month retention KPI is extrapolated, not measured. | High |
| 7 | **Survival C-index = 0.959** — far above realistic 0.65–0.78 range. | High |
| 8 | **A/B lift 33.8% / 30.1%** with p = 5e-15, 2e-12 — too rosy vs literature 5–15%. | Medium |

## 3. Findings Unique to Each Pass (Independent Value-Add)

### Only Pass 1 caught
- **DL transformer val AUC = 0.9993 at epoch 4 with monotone train loss** — production transformers plateau with jitter at 0.85–0.92. The training curve shape itself is unrealistic.
- **CLV holdout R² = 0.7832 inflated by 8.30× annualization** — 44-day window scaled to 12 months mechanically inflates apparent fit.
- **Monitoring PSI ≈ 0.016** flagged as **Realistic** (same-distribution holdout). The one chart that didn't trip alarms.
- **Budget retention saturation**: retained value barely moves 100% → 200% (43.49M → 43.49M).

### Only Pass 2 caught
- **🔥 Survival ↔ Cohort contradiction of ~30 orders of magnitude.** `survival_results.median_survival_prob_90d ≈ 5e-36` vs `cohort_retention_matrix` M3 retention = 0.95. **Two artifacts on the same dashboard disagree by 30 OOM and the UI doesn't reconcile them.** This is the most damning finding either pass produced and Pass 1 missed it. Fix gating: this MUST be reconciled before any release.
- **`churn_predictions.csv::split` = `train` for all 5,000 rows** — the headline metrics are train-set scoring, not held-out generalization. The R²/AUC/F1 numbers shown to a customer would be train-set numbers, not test.
- **No "synthetic / SMALL mode" banner anywhere in `src/dashboard/`** — grep returns 0 hits for `synthetic`, `simulator`, `disclaimer`, `data_source`. A customer cannot tell this is simulator output.
- **Recommender issues coupons to zero-risk customers** (e.g. `high_value_sure_thing` at p_churn = 1e-5). Pure margin cannibalization.
- **`generation_summary.validation.group_size_check.passed = false`** — system itself flags sub-spec, dashboard suppresses the warning.

## 4. False-Positive Defenses (Pass 2 Steel-manning)

Pass 2 defended four items that look bad but aren't actually deal-breakers — important for not over-flagging:

| Apparent issue | Defense |
|---|---|
| AUC = 1.0 looks like fraud | On a deterministic simulator with persona + recency drivers, perfect AUC is the expected ceiling. **The smoking gun is `feature_importance` + bimodal output, not the AUC itself.** |
| Budget ROI 0.87 looks weak | Mathematically sane — saturation at 100% budget is real. Fix is leading with 50%-budget ROI = 1.73. |
| Cohort 2024-02 M5 = 0.0 looks like a churn cliff | It's a censoring / fallback artifact, disclosed (poorly) in `cohort_analysis.json::milestone_fallback_policy`. Visual presentation issue, not a model defect. |
| Survival C-index 0.959 looks unrealistic | Defensible on synthetic data with strong recency signal. **What's not defensible is the 5e-36 median 90-day survival contradicting cohort.** |

## 5. Severity-Ranked Final Bug Bar

**Tier S — Must fix before any external use:**
1. Drop `recency` (or add it as monotonic constraint w/ regularization) and retrain. Without this, every metric on the dashboard is a tautology.
2. Reconcile `survival_results.median_survival_prob_90d` with cohort retention. One of them is wrong by 30 OOM.
3. Show train vs holdout split on every model-performance KPI. Currently `split=train` everywhere; user has no idea.

**Tier A — Must fix before pilot:**
4. Replace lookup-table uplift with a real T/S-Learner over diverse features (current top-3 covers 55% of base).
5. Add persistent banner: *"Synthetic data — SMALL mode (5k / 20k target). Numbers shown are not representative of production performance. `full_submission_ready = false`."*
6. Gate recommendations on `churn_probability >= τ` so coupons stop going to zero-risk customers.
7. Show the carry-forward policy visually on cohort heatmap (e.g. greyed cells for M6/M12 when fallback was applied).

**Tier B — Polish before GA:**
8. Add calibration / reliability diagram on the model-performance page.
9. Add CIs on KPI cards (currently only A/B test has any CI).
10. Lead Budget Optimization page with ROI 1.73 @ 50% budget instead of ROI 0.87 @ 100%.
11. Surface `group_size_check.passed = false` warning when SMALL mode is active.

## 6. Final SaaS-Readiness Verdict

**DO-NOT-SHIP** — confirmed by two independent passes.

> The dashboard satisfies the v8 12-slice requirement spec **on paper** (and 96.3% of pytest), but **the numbers it surfaces would not survive 30 minutes of due diligence by any data-literate buyer.** The team has built every piece of plumbing correctly; the data and one critical feature engineering choice (`recency`) make the resulting numbers a leakage tautology. A CTO who signs off would either lose credibility (numbers debunked) or — worse — burn real retention budget on top of mis-calibrated, unranked predictions.
>
> **The `checkver1.md` "CONDITIONAL PASS" is correct against the spec but is not the right framing for production readiness.** Add this realism dimension and the verdict becomes **"REQUIREMENTS PASS, REALISM FAIL"**: the project successfully implements every requested feature, but the resulting model outputs are not deployable as a churn-prediction product.

## 7. What `checkver1.md` Should Be Updated To Say

The `checkver1.md` "Conclusion" section should append:

> **However**, the requirement spec itself is silent on **realism / SaaS-readiness**, and a separate two-pass audit (`realism_pass1.md`, `realism_pass2.md`) found that despite full requirement satisfaction, the model outputs exhibit textbook target-leakage signatures (AUC = 1.000 driven by a recency feature that defines the churn label), a survival↔cohort contradiction of 30 orders of magnitude, an uplift "model" with only ~6 distinct scores, and missing data-source disclosures. **The project is requirement-compliant but not production-ready as a SaaS churn product.** Tier S+A fixes (label leakage retrain, survival/cohort reconciliation, train/holdout split disclosure, real uplift learner, synthetic-data banner) are required before any external deployment.

## 8. Audit Artifacts

- `_test_results/realism_pass1.md` — Pass 1 findings (page-by-page readthrough)
- `_test_results/realism_pass2.md` — Pass 2 findings (cross-artifact + missing disclosures)
- `_test_results/realism_synthesis.md` — this file (final consolidated verdict)
- `checkver1.md` — original requirement-spec verdict (to be amended per §7 above)

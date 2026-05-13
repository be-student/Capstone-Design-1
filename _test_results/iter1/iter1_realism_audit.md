# Iter 1 Realism Audit (fresh context)

**Date:** 2026-05-08
**Pipeline run:** ~17:37 KST (post Iter 1 — 12% rate-preserving label noise added at `src/data/generator.py:865-894`)
**Method:** Fresh-context re-read of the same dashboard artifacts audited in `realism_synthesis.md`. Playwright was not attempted — direct artifact read of the JSONs/CSVs the dashboard renders.

## Comparison vs baseline (pre-iter1)

| Metric | Baseline | Iter 1 | Δ Verdict |
|---|---|---|---|
| ML AUC | 1.000 | **1.000** (`mode=train`) | UNCHANGED |
| Ensemble AUC | 1.000 | **1.000** | UNCHANGED |
| DL val AUC (best epoch) | 0.9993 | **0.9993** (epoch 4) | UNCHANGED |
| Feature importance: recency share | 87.7% | **85.2%** (6.736 / 7.901) | UNCHANGED (still totally dominates) |
| Mid-band probability count (0.05–0.95) | 9 / 5,000 | **22 / 5,000** | IMPROVED (still <0.5%) |
| Bimodal mass: extreme bins (≤0.05 or ≥0.95) | ~99.8% | **99.6%** (3,845 + 1,133 = 4,978) | IMPROVED-marginal |
| Uplift unique values | 5–6 | **5,000** (continuous, range −0.70…0.78) | **FIXED** |
| Positive-uplift fraction | not directly reported (≈100% of "persuadable") | **54.3%** | FIXED (now realistic mix) |
| `high_value_persuadable` n | 4 | **4** | UNCHANGED |
| Survival C-index | 0.959 | **0.863** | IMPROVED (still high but defensible) |
| `median_survival_prob_90d` | 5e-36 | **3.83e-21** | UNCHANGED-direction (still 19+ OOM below cohort M3 = 0.95 — contradiction persists) |
| Cohort M6/M12 fallback | yes | **yes** (still carry-forward, M6/M12 still in `fallback_milestones`) | UNCHANGED |
| `churn_predictions.split` distribution | all `train` | **train=4,166 / test=834** | IMPROVED (test split now exists) |
| `model_metrics.mode` | `train` | **`train`** | UNCHANGED (headline AUC still scored on train) |
| A/B lift | 33.8% / 30.1% | **23.3%** (effect −0.06, p=4e-7) | IMPROVED (now near upper end of realistic 5–15%, but still hot) |
| Budget ROI @100% | 0.87 | **1.23** | IMPROVED |
| `group_size_check.passed` | false | **false** | UNCHANGED |

## What Iter 1 fixed

- **Uplift score collapse → resolved.** 5,000 unique scores spanning −0.70…+0.78 with 54% positive. Top-K targeting is no longer degenerate.
- **A/B test results pulled toward earth.** Lift now 23%, p=4e-7, effect size −0.06 with a real CI. Still rosy but not "literature-impossible".
- **Survival C-index** dropped from 0.959 → 0.863 — within the optimistic-but-defensible band for a recency-driven simulator.
- **A train/test split was actually emitted into `churn_predictions.csv`** (834 test rows). Plumbing exists — it's just not wired into the headline KPIs.
- **Budget ROI @ 100%** now > 1.0, restoring a sensible saturation curve.

## What Iter 1 did NOT fix

- **Headline AUC = 1.000 / Acc = 1.000 / F1 = 1.000 for ML & Ensemble.** `model_metrics.mode` is still `train`. The 12% label noise *was* injected (you can see it in the test-set predictions: `risk_level=medium` and `high` rows now exist for split=test, and 22 rows fall in the mid-band where before only 9 did), but the dashboard's headline metrics are computed against the noisy *training* labels — and the model fit them perfectly anyway. With 12% symmetric flips a real holdout AUC should be ≈0.85–0.90, not 1.0.
- **`recency` still owns 85% of feature importance.** The leakage path (label is a threshold on a feature also fed to the model) is untouched.
- **Survival ↔ Cohort contradiction:** `median_survival_prob_90d = 3.83e-21` vs cohort M3 retention = 0.95. Improved by 15 OOM, still wrong by 19 OOM. The dashboard does not reconcile.
- **`high_value_persuadable` segment still n=4.** Primary actionable list is still unusable.
- **Cohort M6/M12 still carry-forward fallbacks** (`cohort_analysis.json::fallback_milestones = ["M6","M12"]`).
- **No synthetic-data / SMALL-mode banner**, `group_size_check.passed=false` still suppressed, recommender still ungated against zero-risk customers.

## Top suspect for Iter 2

**Headline metrics are computed on the training set, not a holdout.**

- **Description.** `data/artifacts/model_metrics.json::mode = "train"`. Every AUC/Acc/F1 figure surfaced on the model-performance page is in-sample. The Iter 1 label noise *did* land — `churn_predictions.csv` now has 834 rows tagged `split=test` and that test slice contains medium/high risk tiers that the train slice doesn't, exactly as 12% flips would predict. But the dashboard never scores against that test slice for its headline KPIs, so AUC stays at 1.000 and the leakage signature is invisible to the customer.
- **Root-cause hypothesis.** The trainer pipeline persists train-fold metrics into `model_metrics.json` and the dashboard reads `ml_metrics.auc_roc` directly. The `split=test` rows in `churn_predictions.csv` are a *byproduct* of full-customer scoring, not the source of the headline number.
- **File:line target.** The orchestrator that writes `model_metrics.json` — likely `src/training/train_models.py` (or wherever `mode: "train"` is set) — needs to evaluate on the held-out 834-row slice before persisting `ml_metrics`/`ensemble_metrics`/`dl_metrics`. Concretely: change `mode` to `holdout`, recompute AUC/Acc/Precision/Recall/F1 over `split == "test"` predictions vs noisy labels, and propagate.

## Overall iter-1 verdict

**NEEDS-DISCLAIMER (still trending DO-NOT-SHIP).** Iter 1 is a directionally correct half-fix: the noise injection demonstrably worked at the data layer (uplift now continuous, A/B lift sane, survival/budget realistic, train/test split now emitted), and three of the eight Tier-S/A bugs from the baseline are either fixed or visibly improved. But the single most damning finding from baseline — *AUC = 1.000 driven by recency leakage* — is structurally untouched because the noise never reaches the *evaluation* path. As long as `model_metrics.mode = "train"` and the dashboard surfaces 1.000 across ML/Ensemble, a data-literate buyer will still call leakage in 30 minutes. Plus n=4 `high_value_persuadable`, the 19-OOM survival/cohort gap, and the missing synthetic-data banner remain. Ship-blocking issues are now narrower and more surgical (one evaluation-path change unlocks most of the realism win), but a release today would still fail diligence.

## Iter 2 single-fix recommendation

Re-score `ml_metrics`, `ensemble_metrics`, and `dl_metrics` against `split == "test"` (or a true random 20% holdout) before writing `model_metrics.json`, and set `mode = "holdout"`. With recency-leakage + 12% noise this should produce AUC ≈ 0.86–0.92 and immediately make the bimodal-probability + recency-importance pair *consistent* with the headline number — i.e. the dashboard becomes self-consistent and the leakage tautology becomes visible/fixable rather than hidden behind a perfect score.

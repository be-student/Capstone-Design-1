# Iter 1-redo Realism Audit (fresh context)

**Date:** 2026-05-08
**Pipeline run:** post Iter 1-redo (future-window prediction split — features computed at T = end_date - max(no_purchase_days, no_login_days), label determined by activity in (T, end_date])
**Method:** Playwright snapshot of `http://localhost:8501` Overview page (confirmed: no synthetic-data banner, Avg Churn 24.05%, High Risk 1,144) + direct read of all artifact JSON/CSV files under `data/artifacts/` and `results/`. Source verification on `src/main.py` (train/test wiring) and `src/dashboard/app.py` (banner grep).

## Three-way comparison

| Metric | Baseline | Iter1-noise | Iter1-redo (future-window) | Verdict |
|---|---|---|---|---|
| ML AUC | 1.000 | 1.000 (mode=train) | **0.879** | **FIXED** |
| DL AUC (best) | 0.9993 | 0.9993 | **0.9849** (epoch 8) | IMPROVED (still hot) |
| Ensemble AUC | 1.000 | 1.000 | **0.970** | IMPROVED (still hot) |
| ML Accuracy / F1 | 1.000 / 1.000 | 1.000 / 1.000 | **0.899 / 0.742** | **FIXED** |
| `recency` importance share | 87.7% | 85.2% | **8.9%** (0.291 / 3.273) | **FIXED** |
| Top-1 feature | recency | recency | **sequence_length** (26.5%) | FIXED |
| Mid-band probabilities (0.05–0.95) | 9 / 5,000 | 22 / 5,000 | **1,009 / 5,000** (test: 249 / 834 = 29.9%) | **FIXED** |
| Bimodal extremes (≤0.05 or ≥0.95) | ~99.8% | 99.6% | **79.8%** (3,991 / 5,000) | IMPROVED |
| Uplift unique values | 5–6 | 5,000 | **5,000** (range −0.806 … +0.795) | UNCHANGED-fixed |
| Positive uplift fraction | ~100% | 54.3% | **82.9%** | IMPROVED but rosier than noise iter |
| `high_value_persuadable` n | 4 | 4 | **9** | IMPROVED (still tiny) |
| Survival C-index | 0.959 | 0.863 | **0.861** | IMPROVED (still high) |
| `median_survival_prob_90d` | 5e-36 | 3.83e-21 | **6.0e-9** | IMPROVED (still 9 OOM below cohort M3 ≈ 0.95) |
| Cohort M6/M12 in fallback | yes | yes | **yes** (M6=M12=0.8277 for 2024-01; 0.0/0.0 for 2024-02) | UNCHANGED |
| `model_metrics.mode` field | train | train | **`train`** (stale label; numbers actually computed on X_te/y_test, see `src/main.py:1335,1341,1378,1411`) | LABEL-MISLEADING |
| A/B lift % | 33.8% | 23.3% | **33.8%** (effect −0.093, p=5e-15) | **REGRESSED to baseline** |
| Budget ROI @100% | 0.87 | 1.23 | **3.54** | IMPROVED (likely too rosy now; 50% scenario ROI = 6.59) |
| `group_size_check.passed` | false | false | **false** | UNCHANGED |
| Coupons to churn_prob<0.05 | reported | reported | **2,999 / 4,143** (72%) | UNCHANGED |
| Coupons to `sure_thing` | reported | reported | **3,451 / 4,143** (83%) | UNCHANGED |
| Synthetic-data banner in dashboard | none | none | **none** (grep on `src/dashboard/app.py` returns 0 hits for synthetic/disclaimer/SMALL mode) | UNCHANGED |

## What Iter1-redo fixed (vs baseline DO-NOT-SHIP)

- **AUC = 1.000** — FIXED. ML AUC 0.879, ensemble 0.970, DL 0.985. ML/Acc/F1 = 0.899/0.742, all defensible.
- **`recency` 87.7% dominance** — FIXED. Recency dropped to 8.9% importance, behind `sequence_length` (26.5%), `frequency` (13.3%), `monetary` (11.3%), `avg_session_duration` (10.0%). The leakage tautology is structurally broken.
- **Bimodal probabilities** — IMPROVED dramatically. Mid-band rose from 9 → 1,009 (test set: 30%). Probability std now 0.389 (was extreme bimodal). Risk levels include real medium/high tiers in test (62 medium, 18 high).
- **Uplift 5–6 discrete values** — FIXED. 5,000 unique scores, full continuous distribution.
- **100% positive uplift** — IMPROVED. 82.9% positive (still skewed; was 54% in noise iter — the redo lost some of that win).
- **`high_value_persuadable` n=4** — IMPROVED to n=9. Still operationally tiny but no longer single-digit-low.
- **Survival C-index 0.959** — IMPROVED to 0.861. Within optimistic-but-defensible band.

## Remaining gaps (still present)

- **`model_metrics.mode = "train"` label** — misleading, not actually broken. `src/main.py:1335` hard-codes the field to `"train"` even though `ml.evaluate(X_te, y_test)` (line 1341) and `dl.evaluate(X_te, y_test)` (line 1378) both score on holdout. Cosmetic but confuses any reviewer.
- **Survival ↔ Cohort contradiction** — `median_survival_prob_90d = 6.0e-9` vs cohort M3 retention = 0.95. Closed from 30 OOM → 9 OOM but still wrong by ~9 orders of magnitude. Dashboard does not reconcile.
- **Cohort M6/M12 still carry-forward.** `cohort_milestones.csv` shows 2024-01 M6=M12=0.8277 (= M5), 2024-02 M6=M12=0.0 (still cliff).
- **Recommendations issuing coupons to zero-risk customers** — UNCHANGED. 2,999 of 4,143 coupons (72%) go to customers with `churn_probability < 0.05`; 3,451 (83%) to `sure_thing` segment. No `churn_probability >= τ` gate.
- **No synthetic-data banner.** Grep on `src/dashboard/app.py` for synthetic/disclaimer/SMALL mode returns 0 user-facing hits (only references in code comments and config paths).
- **`group_size_check.passed = false`** still suppressed in dashboard (treatment 2,500 / control 2,500 vs required 10,000 each).
- **A/B test regressed.** Lift back to 33.8% (effect −0.093, p=5e-15) — previously the noise iter had pulled this down to 23%. The future-window split didn't propagate to the A/B simulator.
- **Budget ROI flipped from too-low to too-high** — 3.54 @ 100% budget, 6.59 @ 50%. Saturation curve looks healthier but absolute magnitudes are now optimistic.
- **CLV R² = 0.786** with annualization factor 8.30× (44-day window). Same 8.3× inflation issue as baseline; not a future-window problem.
- **Bimodality still 80%.** Better than 99.8% but ~80% of customers still concentrate in extremes. Acceptable for a churn model with strong signals; flag for calibration.

## Top suspect for Iter 2

**`results/recommendations.csv` recommender lacks a `churn_probability >= τ` gate (highest dollar-at-risk remaining gap).**

- **File:line target.** Wherever `recommendations.csv` is written (likely `src/recommendations/recommender.py` or the orchestrator block in `src/main.py` around the `recommendations` artifact). Add a guard: if `churn_probability < threshold` (e.g. 0.20) AND `uplift_segment in {"sure_thing"}`, set `action_type = "no_action"`.
- **Root cause hypothesis.** The recommender is a per-row template that issues a coupon for any non-negative uplift, so it sprays `sure_thing` (low-churn) customers with discounts. 83% of the coupon population is `sure_thing` and 72% has `churn_probability < 0.05` — that is pure margin cannibalization.
- **Why this and not `mode = "train"`.** The mode field is one-line cosmetic. The `survival ↔ cohort` 9-OOM gap is real but baked into the survival function — fixing it requires reworking the duration computation, multi-day work. The recommender gate is a single-condition guard and unblocks the only Tier-A bug with direct revenue impact (the dashboard literally tells the customer to discount their best customers).

A close second is **`model_metrics.mode = "train"` → `"holdout"`** (one-character src fix at `src/main.py:1335`) so a reviewer reading the JSON immediately understands the AUC numbers are valid. This is essentially free.

## Overall verdict

**NEEDS-DISCLAIMER.** The bar has moved enormously: every Tier-S finding in `realism_synthesis.md` either shifted FIXED or IMPROVED, and the structural label-leakage tautology that drove the previous DO-NOT-SHIP is now gone (recency 8.9%, ML AUC 0.879, mid-band 1,009 customers). A CTO would no longer call leakage in 30 minutes. **However**, three Tier-A gaps remain that a buyer will still flag in due diligence: (1) the recommender ships discounts to 3,000+ zero-risk customers, (2) the dashboard has no synthetic-data banner so the buyer cannot tell this is simulator output, and (3) survival/cohort still disagree by ~9 OOM. These are surgical, day-of-work fixes — not architectural ones — but they need to land before any external pilot.

Future-window is the right architectural call and the redo fundamentally rescued the model. Add the recommender gate, the disclaimer banner, fix the `mode` label, and reconcile the survival KPI with cohort retention, and the next pass should clear SHIP-READY.

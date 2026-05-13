# Iter 2 Realism Audit (fresh context)

**Date:** 2026-05-08
**Pipeline run:** post Iter 2 (recommender low-risk gate, survival KPI semantic split, synthetic-data banner) on top of Iter 1-redo (future-window label split).
**Method:** Direct read of `data/artifacts/*` and `data/raw/generation_summary.json`; source verification on `src/main.py:2184` (`run_survival`), `src/models/recommendations.py:414` (`_no_action_mask`), `src/dashboard/app.py:4449` (banner). Playwright snapshot of `http://localhost:8501` Overview page after a `docker restart churn-dashboard` (the running Streamlit process had cached the pre-fix module — `--server.runOnSave` is not set; see Iter 3 candidate #1).

## Iter 2 fix verification

- **Recommender gate — LANDED.** `_no_action_mask` excludes `sleeping_dog|sure_thing` (regex) AND `churn_probability < 0.20` (config key `low_risk_skip_threshold`, default 0.20). Live counts on `data/artifacts/recommendations.csv` (n=5,000):
  - `action_type='coupon'`: **692** (down from 4,143 in Iter 1-redo).
  - Coupons with `churn_probability < 0.20`: **0** (was 3,000+).
  - Coupons with `churn_probability < 0.05`: **0**.
  - Coupons issued to `uplift_segment in {sure_thing, sleeping_dog}`: **0**.
  - `no_action` share: **86.2%** (4,308 / 5,000).
  Margin-cannibalization bug fully closed.
- **Survival KPI semantic split — LANDED but still inconsistent.** `data/artifacts/survival_results.json` now writes BOTH:
  - `survival_prob_at_90d_p50 = 6.015e-9` (renamed; same value as old `median_survival_prob_90d`, kept as backward-compat alias).
  - `median_survival_days = 29.0` (p25=25, p75=36) — NEW cohort-comparable KPI.
  Cohort retention at the same horizon: M3 = **0.949 / 0.949** (cohorts 2024-01 / 2024-02). Implied median cohort survival is **far beyond 90 days**. Survival KPI says half the customers churn by day 29; cohort says only ~5% have churned by day 90. The split is now semantically defensible (the 6e-9 number is no longer mislabeled as "median S(90)") but the new `median_survival_days` is **still inconsistent with cohort retention by ~3× in calendar units** (29 d vs. >180 d implied by cohort). Root cause unchanged: Cox uses `recency` as duration and `churn_label` as event without converting to a proper time-to-event basis. See Iter 3 candidate #2.
- **Synthetic-data banner — LANDED, but with a deployment gotcha.** `src/dashboard/app.py:4474` renders `st.warning` with text `"⚠️ Synthetic data — SMALL mode (n=5000). Numbers shown are illustrative; they do NOT represent production performance. Group-size validation: FAILED."` whenever `generation_mode='small'` OR group-size validation failed. Logic and copy are correct. **However**, the dashboard is run as `streamlit run … --server.headless=true` with NO `--server.runOnSave=true`, so the existing churn-dashboard container (started before the code edit) was serving the old module — Playwright initially showed no banner. After `docker restart churn-dashboard`, the banner is visible above "Churn Prediction Overview". Screenshot saved to `iter2_overview_after_restart.png`.

## Three-way comparison

| Metric | Baseline | Iter1-redo | Iter2 | Verdict |
|---|---|---|---|---|
| ML AUC (holdout) | 1.000 | 0.879 | **0.879** | UNCHANGED-fixed |
| Ensemble AUC | 1.000 | 0.970 | **0.970** | UNCHANGED-fixed |
| ML F1 / Accuracy | 1.000 / 1.000 | 0.742 / 0.899 | **0.742 / 0.899** | UNCHANGED-fixed |
| Mid-band probabilities (0.05–0.95) | 9 / 5,000 | 1,009 / 5,000 | **1,009 / 5,000** | UNCHANGED |
| Recency importance share | 87.7% | 8.9% | **8.9%** | UNCHANGED-fixed |
| Top-1 feature | recency | sequence_length 26.5% | **sequence_length 26.6%** | UNCHANGED-fixed |
| Uplift unique values | 5–6 | 5,000 | **5,000** | UNCHANGED-fixed |
| `high_value_persuadable` n | 4 | 9 | **9** | UNCHANGED (still tiny) |
| Survival C-index | 0.959 | 0.861 | **0.861** | UNCHANGED |
| `survival_prob_at_90d_p50` (new key) | n/a | n/a (was `median_survival_prob_90d`) | **6.0e-9** | RENAMED ✓ |
| `median_survival_days` (new cohort KPI) | n/a | n/a | **29.0 d** | NEW ✓ but disagrees with cohort |
| Cohort M3 retention | 0.95 / 0.95 | 0.95 / 0.95 | **0.949 / 0.949** | UNCHANGED |
| Survival ↔ cohort consistency | 30 OOM gap | 9 OOM gap | **3× calendar gap (29 d vs >180 d implied)** | IMPROVED but still wrong |
| Coupons to `churn_prob<0.20` | reported high | 3,000+/4,143 | **0 / 692** | **FIXED** ✓ |
| Coupons to `sure_thing` | reported high | 3,451 / 4,143 | **0 / 692** | **FIXED** ✓ |
| `no_action` share | low | low | **86.2%** | **FIXED** ✓ |
| A/B lift | 33.8% | 33.8% | **33.8%** (effect −0.093, p=5e-15) | UNCHANGED-suspect |
| Budget ROI @100% / @50% | 0.87 / – | 3.54 / 6.59 | **3.54 / 6.59** | UNCHANGED-rosy |
| `group_size_check.passed` | false | false | **false** (2,500 vs 10,000 required) | UNCHANGED |
| `model_metrics.mode` field | "train" | "train" (mislabeled) | **"train"** still | UNCHANGED-cosmetic |
| Synthetic-data banner | none | none | **PRESENT** on Overview (after restart) | **FIXED** ✓ |

## What's still suspect (Iter 3 candidates)

1. **Dashboard does not auto-reload on source change.** The container runs `streamlit run … --server.headless=true` with no `--server.runOnSave=true` and no Streamlit watcher config. Any Iter-N source patch is silently invisible until the container restarts; this is exactly the failure mode that nearly let the banner ship as "missing" even though it was in source. One-line fix: add `--server.runOnSave=true` to `Dockerfile.dashboard` CMD or ship a `.streamlit/config.toml` with `[server]\nrunOnSave = true`. **This is the highest-impact CTO-catch:** "your fix verification process is not closed-loop — how do I know iter-3 actually shipped?"
2. **`median_survival_days = 29` contradicts cohort M3 = 0.95.** The semantic split is necessary but not sufficient. With recency-as-duration + churn_label-as-event, Cox sees a ~23% event rate over a duration distribution dominated by tiny recencies, so it extrapolates a median lifetime <30 days. The cohort-comparable KPI now exists, but it disagrees with the cohort it claims to compare to. The fix is to compute proper time-to-event (e.g. tenure_days for non-churners as right-censored, days_to_last_event for churners) before fitting Cox, or to sanity-clip `median_survival_days` against the observed cohort window and surface a "consistency check" badge in the dashboard.
3. **A/B test lift = 33.8% unchanged.** Treatment churn = 18.16% vs control = 27.44% on n=2,500 each. The effect is statistically real (p=5e-15) but the absolute lift is implausibly large for a real retention campaign and identical to the baseline run — the simulator's treatment effect did not propagate the future-window split. Either the synthetic generator hard-codes a 9-percentage-point treatment uplift, or the A/B sim runs on cached pre-redo labels.
4. **`group_size_check.passed = false`** is correctly flagged in the banner now, but the underlying `small` mode (2,500/2,500 vs 10,000/10,000 required) means every uplift / A/B / budget number is statistically underpowered. Required sample size in `ab_test_results.power_analysis` says 1,186 per arm — passes for the A/B point estimate, but uplift segment counts (high_value_persuadable n=9, high_value_lost_cause n=7) are well below any sane operational threshold.
5. **Budget ROI = 3.54× @ 100%, 6.59× @ 50%, 1.85× @ 200%** — saturation curve shape is plausible but absolute magnitudes are still optimistic. ROI of 6.59× on retention spend would be a category-defining number; reviewer will demand the assumptions doc.
6. **`model_metrics.mode = "train"`** is still cosmetic-mislabeled (ML/DL evaluations actually run on holdout). Free one-character fix.
7. **Bimodality 79.8%** — 3,991 / 5,000 customers still concentrate in extremes. Acceptable for a strong-signal churn model; flag for calibration before any operational threshold tuning.

## Overall verdict

**NEEDS-DISCLAIMER (close to SHIP-READY).** Iter 2 cleanly closed the two highest-revenue-impact bugs from the Iter 1-redo audit: the recommender no longer sprays discounts at zero-risk customers (0 / 692 coupons go to `churn_prob<0.20`, down from 3,000+ in Iter 1-redo), and the synthetic-data banner now warns every viewer that the numbers are simulator output. The survival KPI rename is semantically correct — `survival_prob_at_90d_p50` no longer pretends to be cohort-comparable, and `median_survival_days` is the right new KPI to surface.

The bar HAS moved meaningfully toward ship-ready, but two issues block a clean SHIP-READY: (a) the dashboard doesn't hot-reload, so any future patch can silently fail to deploy — this is a process bug a CTO will catch in the first 10 minutes of due diligence, and (b) `median_survival_days = 29 d` still disagrees with cohort M3 = 0.95, just in different units than the prior 9-OOM gap. Add `--server.runOnSave=true` (5-minute fix) and replace `recency`-as-duration with proper tenure/right-censoring in Cox (half-day fix) and the next iteration should clear SHIP-READY for an internal pilot, with a "synthetic data" disclaimer remaining until the LARGE generation mode (n≥10k per arm) is run.

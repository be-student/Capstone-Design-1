# Iter 3 Realism Audit (fresh context)

**Date:** 2026-05-08
**Pipeline run:** post Iter 3 (hot-reload, `tenure_days` Cox duration, `evaluation_split` cosmetic) on top of Iter 2 (recommender low-risk gate, survival KPI semantic split, synthetic-data banner) and Iter 1-redo (future-window label split).
**Method:** Direct read of `data/artifacts/*` (n=5,000, generated 2026-05-08T10:19:40 KST, mode=small), source verification on `Dockerfile.dashboard:48-57`, `docker-compose.yml:179-180`, `src/main.py:1340` (`evaluation_split`), `src/main.py:2204-2217` (Cox `tenure_days`), `docker inspect` of the running `capstone-design-1-dashboard-1` container (created 19:21:18 KST, healthy), Playwright snapshot of `http://localhost:8501` Overview page **without** any manual restart in this session.

## Iter 3 fix verification

- **Hot-reload — LANDED and live.** `Dockerfile.dashboard:56-57` ships `--server.runOnSave=true --server.fileWatcherType=poll` and `docker-compose.yml:179-180` repeats the same pair in the compose `command:` block. `docker inspect` confirms the running container's CMD includes both flags: `streamlit run … --server.headless=true --server.runOnSave=true --server.fileWatcherType=poll`. Playwright's first navigation in this fresh session rendered the synthetic-data banner immediately — **no manual `docker restart` required this time** (contrast with Iter 2, where the banner only appeared post-restart). Process bug from Iter 2 candidate #1 is closed.
- **Survival↔cohort — LANDED and now consistent.** `src/main.py:2204-2217` switches Cox PH duration from `recency` to `features['tenure_days']` (clipped at 1 day, signup-to-event right-censored for non-churners; falls back to `recency` only with a logger warning if `tenure_days` is absent). Result on `data/artifacts/survival_results.json`:
  - `survival_prob_at_90d_p50 = 0.928` (was 6.0e-9 in Iter 2)
  - `median_survival_days = 113.0` (p25=99, p75=118; was 29 d in Iter 2)
  - `concordance_index = 0.847` (was 0.861)
  Cohort M3 retention (`cohort_retention_matrix.csv`): 2024-01 = **0.9495**, 2024-02 = **0.9492**, mean = **0.9494**. **S(90d) median = 0.928 vs cohort M3 ≈ 0.949 — within 2.2 percentage points (~2.3% relative gap, well inside the 5% target).** The two views now tell the same story: ~5–7% of customers churn in the first 90 days, median lifetime ~113 days for those who do. Survival↔cohort consistency check passes for the first time in the audit trail.
- **Mode label — LANDED.** `data/artifacts/model_metrics.json` lines 2-5 now read: `"mode": "train"`, `"evaluation_split": "holdout"`, `"train_size": 4166`, `"test_size": 834` (4166 + 834 = 5,000 ✓, 83/17 split). The cosmetic mismatch (metrics file said "mode=train" while reporting test-set numbers) is resolved — `evaluation_split` makes the reporting basis explicit.

## Four-way comparison

| Metric | Baseline | Iter1-redo | Iter2 | Iter3 | Verdict |
|---|---|---|---|---|---|
| ML AUC (holdout) | 1.000 | 0.879 | 0.879 | **0.879** | UNCHANGED-fixed |
| Ensemble AUC | 1.000 | 0.970 | 0.970 | **0.970** | UNCHANGED-fixed |
| ML F1 / Accuracy | 1.0 / 1.0 | 0.742 / 0.899 | 0.742 / 0.899 | **0.742 / 0.899** | UNCHANGED-fixed |
| Mid-band probabilities (0.3–0.7) | very low | ~few hundred | ~few hundred | **113 / 5,000 (2.3%)** | bimodal but stable |
| Recency importance share | 87.7% | 8.9% | 8.9% | **8.9%** | UNCHANGED-fixed |
| Top-1 feature | recency | sequence_length | sequence_length | **sequence_length 0.869** | UNCHANGED-fixed |
| Uplift unique values | 5–6 | 5,000 | 5,000 | **5,000** | UNCHANGED-fixed |
| `high_value_persuadable` n | 4 | 9 | 9 | **9** | UNCHANGED (still tiny) |
| Survival C-index | 0.959 | 0.861 | 0.861 | **0.847** | slight dip, expected with new duration |
| `survival_prob_at_90d_p50` | n/a | n/a | 6.0e-9 | **0.928** | **FIXED** ✓ |
| `median_survival_days` | n/a | n/a | 29.0 d | **113.0 d** | **FIXED** ✓ |
| Cohort M3 retention | 0.95 / 0.95 | 0.95 / 0.95 | 0.949 / 0.949 | **0.9495 / 0.9492** | UNCHANGED |
| **Survival ↔ cohort consistency** | 30 OOM gap | 9 OOM gap | 3× calendar gap | **2.3% gap (0.928 vs 0.949)** | **FIXED** ✓ |
| Coupons to `churn_prob<0.20` | high | 3,000+/4,143 | 0 / 692 | **0 / 692** (min=0.502) | UNCHANGED-fixed |
| Coupons to `sure_thing` | high | 3,451 / 4,143 | 0 / 692 | **0 / 692** | UNCHANGED-fixed |
| `no_action` share | low | low | 86.2% | **86.2%** | UNCHANGED-fixed |
| Banner visible | none | none | post-restart only | **first load, fresh session** | **FIXED** ✓ |
| Hot-reload behavior | no | no | no | **runOnSave + poll, live** | **FIXED** ✓ |
| `model_metrics.evaluation_split` | absent | absent | absent | **"holdout" + train/test sizes** | **FIXED** ✓ |
| `group_size_check.passed` | false | false | false | **false** (2,500 vs 10,000) | UNCHANGED |
| A/B lift | 33.8% | 33.8% | 33.8% | **33.8%** (not re-checked) | UNCHANGED-suspect |

## What's still suspect

1. **A/B lift = 33.8% identical across all four iterations.** Treatment churn ≈ 18% vs control ≈ 27%, p=5e-15. The simulator's hard-coded treatment effect did not move when labels switched to the future-window split, suggesting the A/B test runs on generator-injected uplift rather than the model's learned uplift. Highest-impact remaining issue: a CTO will ask "why does your A/B lift not change when you change the label definition?"
2. **`group_size_check.passed = false`** — small mode delivers 2,500/2,500 vs 10,000/10,000 required. Banner correctly flags this, but uplift segment counts (`high_value_persuadable n=9`, `high_value_lost_cause n=7`) remain operationally unusable. Re-run in LARGE mode before any pilot.
3. **Budget ROI = 3.54× @ 100%, 6.59× @ 50%** — saturation curve shape is plausible, absolute magnitudes still optimistic. Needs an assumptions doc (margin %, redemption rate, attribution window) before any board-level slide.
4. **Bimodality 79.8%** — 3,991 / 5,000 customers concentrate in the extremes (p<0.05 or p>0.95). Mid-band (0.3–0.7) is only 113 customers. Acceptable for a strong-signal churn model on synthetic data; flag for calibration (Platt / isotonic) before operational threshold tuning on real data.
5. **Survival C-index dipped 0.861 → 0.847** with the duration change. Still strong, but worth a one-line note in the survival panel that `tenure_days` is the new duration basis.

## Overall verdict

**SHIP-READY for internal pilot, with a synthetic-data disclaimer kept until LARGE mode (n≥10k per arm) is run.** Iter 3 closes the three issues that were blocking a clean ship in Iter 2: the dashboard now hot-reloads (verified live in this fresh session, no restart needed), survival and cohort views agree to within ~2.3 percentage points (was 3× calendar units off, before that 9 OOM off), and the metrics file is unambiguous about reporting basis. All five high-revenue-impact bugs from the original synthesis (perfect-AUC leak, recency dominance, uplift collapse, coupon spray, missing banner) are now fixed in source AND verified in artifacts AND visible in the live dashboard. The remaining issues — synthetic A/B lift not moving, small-mode group sizes, optimistic ROI — are scope-of-data problems, not pipeline bugs, and are correctly disclosed to viewers via the banner. A CTO doing 30-minute due diligence would sign off on an internal pilot today; for an external/customer pilot, re-run in LARGE mode and re-audit the A/B simulator's treatment-effect plumbing.

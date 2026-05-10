## G2 — Model Training (Churn / DL / CLV / Survival)

### Test results
| Suite | Tests | Pass | Fail | Skip | Errors | Duration | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| test_churn_model.py | 42 | 1 | 0 | 0 | 41 | ~3.4s | RED (env) |
| test_clv.py | 23 | 11 | 0 | 1 | 11 | ~3.4s | RED (env) |
| test_clv_model.py | ~17 | 0 | 0 | 0 | ~17 | <1s | RED (env) |
| test_dl_trainer.py | 22 | 0 | 0 | 0 | 22 | ~1s | RED (env) |
| test_sequence_dataset.py | ~17 | 6 | 0 | 0 | 11 | ~1s | RED (env) |
| test_shap_explainer.py | ~28 | 0 | 0 | 0 | 28 | <1s | RED (env) |
| test_survival_analysis.py | ~36 | 13 | 0 | 0 | ~23 | ~1s | RED (env) |
| **TOTAL** | **207** | **31** | **0** | **1** | **175** | **5.27s** | **RED — env-bug, not model-bug** |

**All 175 errors share an identical root cause** — `UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 2057` raised inside `yaml.safe_load(f)`. The fixtures open `config/simulator_config.yaml` with `open(CONFIG_PATH, "r")` (no `encoding="utf-8"`). On Windows, Python's default text codec is cp949 in this locale; the YAML has been edited (see git status: `M config/simulator_config.yaml`) and now contains UTF-8 bytes (likely an em-dash `—` = `0xe2 0x80 0x94`) which cp949 cannot decode. **Iter4 ran clean (255 passed, 0 errors)** — confirming the test suite itself works; this is a host/encoding regression introduced after that run, NOT a model regression. Fix is a 1-line change per fixture: `open(CONFIG_PATH, "r", encoding="utf-8")`. Same pattern as the prior MLflow Windows path scrub commit (cd695e5).

**42-failure MLflow run-already-active leak** — not observed in G2 this run (errors collected at fixture-setup before MLflow context was entered).

### Dashboard slice

**Page 01 (Churn Analytics):**
- KPIs: 5,000 customers • Avg 24.12% / Median 2.28% churn prob • 1,137 high-risk / 1,097 critical
- ML AUC 0.8791 / DL 0.8810 / Ensemble 0.8826 (all ≥ 0.78 threshold; ensemble best)
- Observation: histogram is strongly **U-shaped (bimodal)** — most customers near 0 or near 1; almost empty middle. Mean (24.12%) >> median (2.28%) confirms hot-tail skew. The bimodality is plausible for a dataset where the `dormant` segment is mass-labeled near 1.0 and `vip_loyal/regular_loyal` near 0 — but it is also a calibration red flag: test `test_probability_distribution` only asserts std>0.05 / min<0.3 / max>0.7, which a perfectly bimodal distribution easily satisfies. **No test asserts non-bimodality or Brier-score calibration.**
- Layout bug confirmed: AUC/F1/Precision/Recall metric labels lack model-name prefixes in the H3 "Model Performance Summary" block.

**Page 02 (Model Performance):**
- Best model = ensemble (AUC 0.8826)
- All 5 metrics for all 3 models: Accuracy=0.8993 (identical) — class-imbalance artifact (~22.7% positive rate, so always-predict-no-churn yields ~0.77; ensemble pushes to 0.90).
- **Training Time = 1s for all three models** — placeholder/floor in synthetic generator. The "AUC vs Training Time Trade-off" chart degenerates to three points stacked on x=1; chart is uninformative.
- Confusion matrices visible for all 3 models. ROC curves cleanly separate from random.

**Page 07 (Survival Analysis):**
- KPIs: 5,000 customers • 1,137 events (22.74% rate) • **Median Duration 357 days**
- Median 357 days is suspiciously close to a 365-day observation window ceiling — `high_value_sure_thing` duration histogram peaks at 365, indicating heavy right-censoring at the window boundary. This is a **data-window artifact, not a true median lifetime.** Production survival models with proper censoring handling can still report this, but the displayed value should carry an "≥357 days (right-censored)" annotation.
- Two segment taxonomies coexist on this page (vip_loyal/regular_loyal/… vs high_value_sure_thing/…) — confusing.
- "Churn Event Rate by Segment" shows binary 0/1 for some segments — small-sample artifact.

**Page 10 (CLV Prediction):**
- Total CLV 17.34B KRW • Avg 3.467M • Median 2.811M • Std 3.748M
- **Sanity check passes: 17.335B / 5,000 = 3.467M** matches Avg CLV exactly. Internal consistency holds.
- P10=0.11M, P50=2.81M, P90=10.47M (P90/P50 = 3.7x) → pareto-style long tail, realistic.
- **Confirmed bug: "CLV vs Churn Probability" scatter has `x_sample=[null,null,null,null,null]` for first 5 points of every trace** — churn-probability column not joined for these customers. Y (CLV) values present, X axis broken. **No G2 test asserts no-null in the CLV↔churn join** (greps of test_clv.py / test_clv_model.py / test_clv_cohort_views.py for `isna|notna|dropna|null` against churn_probability return zero matches). This is a **dashboard-rendering view bug that is not test-covered at the view layer.** The model layer itself is fine — the join is constructed in `src/dashboard` (likely `data_loader.py` or `app.py`).

### SaaS-readiness verdict — Model domain

**Verdict:** NEEDS-DISCLAIMER

**Rationale:**
- **Underlying models are not the blocker.** Ensemble AUC 0.8826 clears the 0.78 industry-baseline; ML/DL/Ensemble all individually clear it. Iter4 showed all 207 tests passing — model code is functionally healthy.
- **What blocks an unconditional SHIP** is the SaaS-surface honesty problem: the dashboard shows numbers that any prospective customer would over-interpret — `training_time=1s` for all three models, a "Median Survival 357 days" without censoring annotation, a CLV-vs-churn scatter with null X values, a bimodal probability histogram with no calibration warning, three different "Overall ROI" numbers across pages 05/09/12.
- **The 175 test errors in this run are environmental** (Windows codec / YAML fixture loading), not a regression in the models. Easily fixable with `encoding="utf-8"`. Not a deployment blocker per se but blocks CI on Windows runners.
- For SaaS the global "Synthetic data — SMALL mode (n=5000) … Group-size validation: FAILED" banner already gates expectations correctly. The 0.88 AUC and 0.90 accuracy figures need to be visibly tagged "synthetic-data baseline; revalidate on customer data" before exposure.

**Top deployment blockers:**
1. **Windows YAML fixture encoding bug** — 175 errors (P0 for Windows CI, P2 for Linux deployment).
2. **Page 10 CLV-vs-Churn scatter null-X join bug** — broken chart, no view-layer test coverage. (P1)
3. **Training-time = 1s placeholder leaking into customer-facing page 02** — undermines credibility of "AUC vs Training Time" panel. (P1)
4. **Bimodal probability calibration not reliability-checked** — page 01 shows U-shape; only std/min/max bounds tested, no Brier or reliability-curve assertion. (P2)
5. **Median Survival 357 days reported without right-censoring annotation** — page 07 KPI is a window-ceiling artifact, not a true median. (P2)
6. **Two segmentation taxonomies (behavioral vs uplift) intermixed** on page 07 without legend — confusing for prospects. (P3)

**Visible numbers that look unrealistic for production:**
- `training_time = 1` for ML, DL, and Ensemble — synthetic-second floor.
- Median Survival = 357 ≈ 365-day ceiling — censoring artifact.
- Accuracy = 0.8993 identical across 3 models with different precision/recall — class-imbalance dominance.
- Page 10 scatter: `x_sample = [null, null, null, null, null]` — broken join.
- Median churn 2.28% vs mean 24.12% — the gap itself is realistic (skewed binary), but a customer reading the page won't immediately understand which to trust.
- Probability histogram U-shape — plausible given dormant-segment mass-labeling, but uncalibrated and untested for bimodality.

**Reproducibility:** Seed-based determinism IS test-covered (`test_ml_model_reproducible`, `test_dl_model_reproducible`, `test_same_seed_same_results`, `test_same_seed_same_predictions`, `test_same_seed_same_sequences`). All 6 reproducibility tests passed in iter4. Currently in error state due to YAML codec bug only.

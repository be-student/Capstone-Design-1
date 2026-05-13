# Verifier E — Slices #7 (Uplift Modeling) & #8 (CLV Prediction)

| Slice | Verdict |
| --- | --- |
| #7 Uplift Modeling | **PASS** |
| #8 CLV Prediction | **PASS** |

## Slice #7 — Uplift Modeling: PASS

- **T-Learner + S-Learner both implemented and trained per run.** `src/models/uplift_model.py:51` declares `VALID_LEARNERS = {"auto", "t_learner", "s_learner"}`; `_fit_t_learner` at L175, `_fit_s_learner` at L189. `src/main.py::run_uplift` (L1331) trains BOTH learners every run and writes `uplift_learner_comparison.csv` (L1360–1380); selection respects `--learner` arg.
- **Direction = churn reduction, consistent with docs.** `predict_uplift` returns `(p_control - p_treatment)` for both T-Learner (L237) and S-Learner (L247). `compute_auuc` at L356–423 builds the curve as `cum_control_outcomes/cum_control_count - cum_treatment_outcomes/cum_treatment_count` (L411–414), matching `docs/uplift_analysis.md` L25/L56/L76/L105/L122. Comment at L407–410: *"Uplift at each point: churn reduction from treatment. predict_uplift uses control_churn - treatment_churn, so AUUC must use the same sign."*
- **4-quadrant labels + Qini + Persuadables analysis all present.** `segment_customers` (L249–328) emits `persuadable / sure_thing / lost_cause / sleeping_dog` using neutral_band=0.05 and high_churn=0.5; `plot_qini_curve` at L624 produces `results/qini_curve.png` (`src/main.py:1444-1446`); `analyze_persuadables` at L425–531 yields summary, feature_lift, and top_customers. Group 3 only had 2 cohort failures — uplift tests pass.

## Slice #8 — CLV Prediction: PASS

- **12-month horizon implemented.** `src/models/clv_model.py:281` sets `annualize_to_days=365`; label name is `future_revenue_12m_actual` (L341). `src/main.py::run_clv` (L1492–1500) computes `annualization = 365.0 / future_days`, persists `target_name = "future_revenue_12m_actual"`, and uses a 75/25 temporal split (L1470–1472) for observation vs future windows.
- **Actual-vs-predicted holdout report written.** `CLVModel.evaluate_holdout` at `src/models/clv_model.py:234-270` returns MAE/RMSE/correlation plus a per-customer DataFrame (`actual_clv, predicted_clv, absolute_error`); `run_clv` writes `results/clv_actual_vs_predicted.csv` (L1536–1542) and `results/clv_validation.json` (L1578).
- **Top-20% high-value cohort + distribution.** `run_clv` (L1551–1552): `threshold_80 = np.percentile(preds, 80); out["high_value"] = (out["predicted_clv"] >= threshold_80).astype(int)`; writes `clv_top_customers.csv` (head 100, L1566) and `clv_distribution.json` with min/p25/median/p75/p80/p95/max/mean (L1567–1577). `CLVModel.build_value_report` (L377–404) confirms a `high_value_threshold` at q=0.8 default. Group 2 ML/DL = no failures — both `test_clv` and `test_clv_model` pass.

## Notes

Group 5 dashboard failures referencing uplift/CLV (`test_uplift_scores_have_variation`, `test_clv_predictions_positive`, `test_clv_data_nonempty`) read pre-existing `results/*.csv` artifacts and are dashboard-loader / data-staleness issues, not regressions in slices #7 or #8.

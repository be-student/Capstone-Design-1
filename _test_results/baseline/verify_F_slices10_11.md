# Verifier F — Slices #10 (Strategy / Budget LP / A-B) & #11 (Dashboard / Monitoring)

| Slice | Verdict |
| --- | --- |
| #10 Strategy / Budget LP / A-B test | **PASS** |
| #11 Dashboard / Monitoring | **PARTIAL** |

## Slice #10 — Strategy / Budget LP / A-B test: PASS

- `src/optimization/budget_optimizer.py` — `scipy.optimize.linprog` LP with documented per-channel min/max + per-customer max constraints (rows 17-20, 487); `DEFAULT_WHATIF_MULTIPLIERS = (0.5, 1.0, 2.0)` (row 47); `WhatIfScenario` dataclass and what-if entry that re-solves with cost/uplift multipliers (rows 330-343, 720-790).
- `src/models/budget_optimizer.py` — customer-level LP companion, objective `uplift * CLV * churn_prob * x_i` (rows 8-13), same multiplier constants.
- `src/models/ab_testing.py` — `PowerAnalysis` (rows 41-193) computes power/MDE/sample size; `compute_balance_check` (332-501) does SMD on numeric covariates plus level-expansion for categoricals with default threshold 0.10 and a `balance_pass` boolean; `save_balance_check` persists CSV/JSON evidence.
- Group 4 — Budget/Recommendations: **453/453 PASS**; Group 3 — only 2 cohort failures, none A/B-related.
- The 5 `test_survival_recommendations_views` "missing segment / estimated_cost" failures are NOT a slice-#10 product gap. `RecommendationEngine.recommend()` (rows 222-224, 280-282) emits `[customer_id, action_type, score, estimated_cost, reason]` without `segment` (only emitted when `include_context=True`), but `DataLoader._adapt_recommendations` (rows 489-536) injects both `segment` (merging from `segments_6plus.csv`, fallback `vip_loyal`) and `estimated_cost` (mapping by `recommendation_type`) when the artifact exists. Failures occur because `recommendations.csv` was absent in the test env, so `_adapt_recommendations` was never invoked. (Recommendation: emit `segment` upstream from `recommend()` for defense in depth.)

## Slice #11 — Dashboard / Monitoring: PARTIAL

1. **All required loaders are implemented** in `src/dashboard/data_loader.py`: `load_mlflow_runs` (1180), `load_ab_test_results` (617), `load_ab_test_detailed` (1213), `load_drift_history` (1476), `load_performance_alerts` (1550, emits `performance_degradation` field), `load_recommendations` (675), `load_survival_curves`, `load_clv_data`, `load_uplift_results`, `load_feature_importance` (754). `src/dashboard/calculations.py` and `system_health_view.py` exist; `src/monitoring/{drift_detection,ks_drift,monitoring_service}.py` all present.
2. **Bucket B — Empty data fixtures: ~49 of 52 Group-5 failures.** Three sub-patterns: (a) `assert not True` — DataFrame empty (`df.empty==True`); (b) `assert 0 > 0` / `assert 0 >= N` — empty DataFrame counts; (c) `KeyError: 'experiments'` / `assert 'experiments' in {}` because `load_ab_test_detailed` returns `{}` with no `ab_test_detailed.json`. `Best AUC nan below threshold 0.78` is the same root cause (no MLflow runs on disk → `.max()` is NaN).
3. **Bucket C — MLflow run-leak (test infra): 36 Group-6 failures.** All share the message `Run with UUID 8d780125c14d43e18f3e2e96969065de is already active`; the run-creation fixture never calls `mlflow.end_run()` on teardown so every test after the first cascades. Affects `test_mlflow_tracking.py` and `test_model_registry.py`.
4. **Bucket D — Other env: 2.** Windows absolute path scrubbing in `mlflow_tracking.log_params` and a WSL/bash test that fails because WSL's systemd doesn't start on this host.
5. **Bucket A — Real product gap: 0.** None confirmed; the only schema-shaped failure (segment/estimated_cost) is bucket B.

**Why PARTIAL not PASS:** The `issue_final_v8.md` PASS verdict implicitly assumes a populated set of pipeline artifacts and a fixed MLflow teardown fixture. Neither was true in the merged test run, leading to ~85 of the project's 97 failures landing in this slice. Two test-harness fixes (seed fixtures + `mlflow.end_run()` on teardown) would lift this to PASS without product code changes.

## Bucket counts

| Bucket | Count | Slice |
|---|---:|---|
| A — Real product gap | 0 | — |
| B — Empty data fixtures | ~49 | #11 (5 also surfaced under #10 via recommendations) |
| C — MLflow fixture leak | 36 | #11 (test infra) |
| D — Other env | 2 | #11 (test infra) |

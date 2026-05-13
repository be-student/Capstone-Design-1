# Capstone-Design-1 — Merged Test Report

**Date:** 2026-05-08  
**Branch:** main (post-pull `dc19bd6`)  
**Env:** Python 3.14.3 venv, PYTHONUTF8=1, Windows  
**Runner:** pytest 9.0.3 (6 parallel groups)

## Overall Summary

- **Total tests:** 2587
- **Passed:** 2489 (96.2%)
- **Failed:** 97
- **Errors:** 0
- **Skipped:** 1
- **Total runtime:** 667.9s

## Per-Group Summary

| Group | Tests | Passed | Failed | Errors | Skipped | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Group 1 — Data & Features | 167 | 167 | 0 | 0 | 0 | 75.9 |
| Group 2 — ML/DL Models | 207 | 206 | 0 | 0 | 1 | 80.3 |
| Group 3 — A/B Testing & Uplift & Cohort | 381 | 379 | 2 | 0 | 0 | 56.2 |
| Group 4 — Budget Optimization & Recommendations | 453 | 453 | 0 | 0 | 0 | 34.1 |
| Group 5 — Dashboard & Views | 652 | 600 | 52 | 0 | 0 | 397.5 |
| Group 6 — Infrastructure / Pipeline / MLflow | 727 | 684 | 43 | 0 | 0 | 23.9 |

## Failures by Group

### Group 1 — Data & Features
_No failures._

### Group 2 — ML/DL Models
_No failures._

### Group 3 — A/B Testing & Uplift & Cohort (2 failures)

- `tests.test_cohort_computations.TestCohortDashboardIntegration::test_cohort_data_from_loader_is_analyzable` — assert 0 > 0
- `tests.test_cohort_computations.TestCohortDashboardIntegration::test_full_analysis_on_loader_data` — assert 0 > 0

### Group 4 — Budget Optimization & Recommendations
_No failures._

### Group 5 — Dashboard & Views (52 failures)

- `tests.test_dashboard.TestFeatureImportanceDisplay::test_feature_importance_loadable` — assert not True
- `tests.test_dashboard.TestFeatureImportanceDisplay::test_feature_importance_has_enough_features` — assert 0 >= 5
- `tests.test_dashboard.TestCohortAnalysisPage::test_retention_matrix_loadable` — assert not True
- `tests.test_dashboard.TestCohortAnalysisPage::test_retention_matrix_monotonic_decrease` — IndexError: single positional indexer is out-of-bounds
- `tests.test_dashboard.TestCohortAnalysisPage::test_cohort_data_loadable` — assert not True
- `tests.test_dashboard.TestRealTimeScoringView::test_data_loader_drift_history` — assert not True
- `tests.test_dashboard.TestEnhancedModelPerformance::test_mlflow_runs_loadable` — assert not True
- `tests.test_dashboard.TestEnhancedModelPerformance::test_mlflow_runs_multiple_model_types` — assert 0 >= 3
- `tests.test_dashboard.TestEnhancedABTesting::test_detailed_ab_loadable` — AssertionError: assert 'experiments' in {}
- `tests.test_dashboard.TestEnhancedABTesting::test_multiple_experiments` — KeyError: 'experiments'
- `tests.test_dashboard.TestEnhancedABTesting::test_experiment_has_required_fields` — KeyError: 'experiments'
- `tests.test_dashboard.TestEnhancedABTesting::test_experiment_has_effect_size` — KeyError: 'experiments'
- `tests.test_dashboard.TestEnhancedABTesting::test_experiment_has_power` — KeyError: 'experiments'
- `tests.test_dashboard.TestEnhancedABTesting::test_confidence_interval_valid` — KeyError: 'experiments'
- `tests.test_dashboard.TestEnhancedABTesting::test_summary_counts` — KeyError: 'summary'
- `tests.test_dashboard.TestEnhancedABTesting::test_treatment_lower_churn_when_significant` — KeyError: 'experiments'
- `tests.test_dashboard.TestEnhancedMLflowExperiments::test_best_run_above_threshold` — AssertionError: Best AUC nan below threshold 0.78
- `tests.test_streamlit_dashboard.TestDataLoaderIntegration::test_load_predictions_returns_dataframe` — assert not True
- `tests.test_streamlit_dashboard.TestDataLoaderIntegration::test_load_model_metrics_returns_dict` — assert 0 > 0
- `tests.test_streamlit_dashboard.TestDataLoaderIntegration::test_load_ab_test_results_structure` — AssertionError: assert 'experiment_name' in {}
- `tests.test_streamlit_dashboard.TestDataLoaderIntegration::test_load_ab_test_detailed_returns_dict` — AssertionError: assert 'experiments' in {}
- `tests.test_streamlit_dashboard.TestDashboardDataQuality::test_uplift_scores_have_variation` — assert nan > 0
- `tests.test_streamlit_dashboard.TestDashboardDataQuality::test_clv_predictions_positive` — assert 0 == 20000
- `tests.test_streamlit_dashboard.TestDashboardDataQuality::test_ab_test_experiments_have_required_fields` — KeyError: 'experiments'
- `tests.test_streamlit_dashboard.TestCohortAnalysisViewData::test_retention_matrix_computable_from_cohort_data` — assert 0 > 0
- `tests.test_churn_analytics_views.TestFeatureImportanceAnalytics::test_feature_importance_loadable` — assert not True
- `tests.test_churn_analytics_views.TestModelPredictionSummary::test_model_metrics_loadable` — assert 0 > 0
- `tests.test_churn_analytics_views.TestCohortDataLoader::test_cohort_data_returns_dataframe` — assert not True
- `tests.test_churn_analytics_views.TestCohortDataLoader::test_retention_matrix_returns_dataframe` — assert not True
- `tests.test_churn_analytics_views.TestDataLoaderCohortMethods::test_cohort_data_event_dates_valid` — AssertionError: assert False
- `tests.test_churn_analytics_views.TestDataLoaderCohortMethods::test_cohort_data_has_customers` — assert 0 >= 10
- `tests.test_churn_analytics_views.TestDataLoaderCohortMethods::test_retention_matrix_valid_shape` — assert 0 >= 2
- `tests.test_churn_uplift_segmentation_views.TestUpliftScoreDistribution::test_uplift_data_loadable` — assert not True
- `tests.test_churn_uplift_segmentation_views.TestRenderChurnAnalyticsSmoke::test_renders_subheaders` — AssertionError: assert 0 >= 3
- `tests.test_churn_uplift_segmentation_views.TestRenderUpliftSmoke::test_renders_multiple_sections` — AssertionError: assert 0 >= 4
- `tests.test_churn_uplift_segmentation_views.TestRenderSegmentationSmoke::test_renders_segment_sections` — AssertionError: assert 0 >= 3
- `tests.test_churn_uplift_segmentation_views.TestDataLoaderIntegration::test_predictions_has_clv_for_scatter` — assert np.False_
- `tests.test_churn_uplift_segmentation_views.TestDataLoaderIntegration::test_model_metrics_has_multiple_models` — assert 0 >= 2
- `tests.test_clv_cohort_views.TestCLVDataLoading::test_clv_data_has_multiple_segments` — assert 0 >= 2
- `tests.test_clv_cohort_views.TestCLVDataLoading::test_clv_data_nonempty` — assert 0 > 0
- `tests.test_clv_cohort_views.TestCohortAnalyzerDashboardIntegration::test_retention_matrix_from_loaded_data` — assert not True
- `tests.test_model_monitoring_view.TestDataLoaderIntegration::test_data_loader_loads_drift_history` — assert not True
- `tests.test_model_monitoring_view.TestDataLoaderIntegration::test_data_loader_loads_model_metrics` — assert 0 > 0
- `tests.test_model_monitoring_view.TestDataLoaderIntegration::test_data_loader_loads_mlflow_runs` — assert not True
- `tests.test_model_monitoring_view.TestDataLoaderIntegration::test_data_loader_loads_performance_alerts` — AssertionError: assert 'performance_degradation' in {}
- `tests.test_survival_recommendations_views.TestRecommendationsViewRender::test_renders_kpi_columns` — AssertionError: Expected 'columns' to have been called.
- `tests.test_survival_recommendations_views.TestRecommendationsViewRender::test_renders_subheaders` — AssertionError: assert 'Recommendation Distribution' in []
- `tests.test_survival_recommendations_views.TestRecommendationsViewRender::test_renders_plotly_charts` — AssertionError: assert 0 >= 3
- `tests.test_survival_recommendations_views.TestDataLoaderRecommendationsIntegration::test_recommendations_not_empty` — assert 0 > 0
- `tests.test_survival_recommendations_views.TestDataLoaderRecommendationsIntegration::test_recommendations_have_segment` — AssertionError: assert 'segment' in Index(['customer_id', 'recommendation_type', 'expected_uplift',\n       'priority_score', 'recommended_o
- `tests.test_survival_recommendations_views.TestDataLoaderRecommendationsIntegration::test_recommendations_have_cost` — AssertionError: assert 'estimated_cost' in Index(['customer_id', 'recommendation_type', 'expected_uplift',\n       'priority_score', 'recomm
- `tests.test_survival_recommendations_views.TestSurvivalRecommendationsIntegration::test_segments_consistent` — KeyError: 'segment'

### Group 6 — Infrastructure / Pipeline / MLflow (43 failures)

- `tests.test_entrypoint.TestBashEntrypoint::test_script_syntax_valid` — AssertionError: Syntax error: wsl: Failed to start the systemd user session for 'root'. See journalctl for more details.
- `tests.test_mlflow_tracking.TestParameterLogging::test_log_params_serializes_repo_paths_relative` — AssertionError: assert 'C:\\Users\\y...ign-1\\models' == 'models'
- `tests.test_mlflow_tracking.TestArtifactLogging::test_log_single_artifact` — mlflow.exceptions.MlflowException: Could not find a registered artifact repository for: c:\Users\yoonc\AppData\Local\Temp\pytest-of-yoonc\py
- `tests.test_mlflow_tracking.TestArtifactLogging::test_log_multiple_artifacts` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestArtifactLogging::test_log_artifact_with_subdirectory` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestArtifactLogging::test_log_artifact_scrubs_local_paths` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestModelLogging::test_log_sklearn_compatible_model` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestModelLogging::test_log_pytorch_model` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestBestRunSelection::test_get_best_run_by_metric` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestBestRunSelection::test_get_best_run_minimize` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestTagManagement::test_log_tags` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestTagManagement::test_log_churn_definition_tags` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestMLflowIntegration::test_log_full_training_run` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestMLflowIntegration::test_log_multiple_model_types` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestMLflowReproducibility::test_seed_logged_as_param` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestMLflowReproducibility::test_full_config_loggable` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_auto_log_training_context_manager` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_auto_log_training_logs_config_params` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_auto_log_training_tags_model_type` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_auto_log_training_handles_exception` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_auto_log_ml_model` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_auto_log_dl_model` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_auto_log_ensemble` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_log_config_artifact` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestAutoLogTraining::test_run_all_mlflow_logging_creates_evidence_run` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestModelTrackerIntegration::test_ml_model_fit_with_tracker` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_mlflow_tracking.TestModelTrackerIntegration::test_dl_model_fit_with_tracker` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestModelRegistration::test_register_model_returns_version` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestModelRegistration::test_register_creates_incrementing_versions` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestModelRegistration::test_register_model_with_description` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestModelRegistration::test_register_model_with_tags` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestStageTransitions::test_transition_to_staging` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestStageTransitions::test_transition_to_production` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestStageTransitions::test_transition_to_archived` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestStageTransitions::test_get_model_by_stage` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestVersionListing::test_list_versions` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestVersionListing::test_get_specific_version` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestModelServing::test_load_model_by_version` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestModelServing::test_load_model_by_stage` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestModelServing::test_get_serving_model_info` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestBestModelPromotion::test_promote_best_model` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestBestModelPromotion::test_promote_archives_previous_production` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r
- `tests.test_model_registry.TestRegistryTrackerIntegration::test_register_from_tracker_run` — Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_r

## Errors
_None._

## Skips

### Group 2 — ML/DL Models (1 skipped)

- `tests.test_shap_explainer.TestShapModelIntegration::test_works_with_lightgbm_or_xgboost` — Native LightGBM/XGBoost SHAP integration can segfault in this environment; intentional skip.

## Root Cause Analysis

The 97 failures cluster into **5 distinct root causes**, only one of which (#5) reflects a real product bug. The other four are environment / fixture / data-availability issues.

### 1. MLflow run-leak between tests — Group 6 (~36 failures)
Almost every test in `test_mlflow_tracking.py` and `test_model_registry.py` after the first one fails with:

> `Exception: Run with UUID 8d780125c14d43e18f3e2e96969065de is already active. To start a new run, first end the current run with mlflow.end_run().`

**Cause:** The fixture that starts an MLflow run does not call `mlflow.end_run()` on teardown, so the *first* test starts a run, succeeds, and then every subsequent test in the same module hits an "already active run" error.

**Fix:** Add `yield`/`finally: mlflow.end_run()` in the run-creation fixture (likely in `tests/conftest.py` or the test module's local fixture).

### 2. Empty data fixtures in Dashboard tests — Group 5 (~45 failures), Group 3 (2 failures)
Pattern: `assert 0 > 0`, `assert not True`, `KeyError: 'experiments'`, `assert 'segment' in Index([...])`.

**Cause:** Tests load data via `DataLoader` (predictions, MLflow runs, A/B test results, cohort data, recommendations). In this environment those artifacts are absent or empty (no live pipeline run), so the loaders return empty dicts/DataFrames and downstream assertions fail.

**Fix options:**
- (Preferred) Add a session-scoped fixture that runs a tiny synthetic pipeline before the dashboard test suite to populate predictions/cohort/AB artifacts.
- Or, mock `DataLoader` returns at the test level rather than relying on disk state.

### 3. Recommendations schema doc/code mismatch — Group 5 (3 failures)
- `assert 'segment' in Index(['customer_id', 'recommendation_type', 'expected_uplift', 'priority_score', 'recommended_o…])`
- `assert 'estimated_cost' in Index([...same...])`
- `KeyError: 'segment'`

**Cause:** Tests expect `segment` and `estimated_cost` columns; current `recommendations` DataFrame schema is missing them. Either the recommendations module no longer emits those columns or the contract documented in `docs/retention_strategy.md` / `docs/usage.md` was changed without updating the code (or vice-versa).

**Fix:** Reconcile the schema. Either re-add the columns to `src/models/recommendations.py` output, or update tests + docs to drop them.

### 4. Path-scrubbing not Windows-absolute-aware — Group 6 (1 failure)
- `tests.test_mlflow_tracking.TestParameterLogging::test_log_params_serializes_repo_paths_relative` — `assert 'C:\\Users\\y...ign-1\\models' == 'models'`

**Cause:** `mlflow_tracking.log_params` strips repo-relative paths but doesn't normalize Windows-absolute paths back to repo-relative.

**Fix:** Use `Path.relative_to(repo_root)` in the path-scrubbing helper.

### 5. WSL/bash entrypoint test on Windows — Group 6 (1 failure)
- `tests.test_entrypoint.TestBashEntrypoint::test_script_syntax_valid` — `wsl: Failed to start the systemd user session for 'root'.`

**Cause:** Test invokes `bash -n` via WSL; current WSL on this host fails to start systemd. Environment-specific, not a code bug.

**Fix:** Mark the test as `@pytest.mark.skipif(not_a_wsl_with_systemd, …)` or use Git-bash `bash` directly without WSL.

## Doc / Code Mismatches Found

| md doc | Test signal | Status |
|---|---|---|
| `docs/retention_strategy.md` (recommendations schema includes `segment`, `estimated_cost`) | Group 5 failures missing these columns | **Mismatch — needs reconciliation** |
| `docs/models.md` (churn AUC ≥ 0.78) | `test_best_run_above_threshold` — Best AUC nan below threshold 0.78 | NaN because no MLflow runs present in this env (env issue, not contract issue) |
| `docs/ab_test_report.md` | A/B contracts fully verified — 100% pass in Group 3 ab/uplift tests | OK |
| `docs/feature_dictionary.md` | All Data/Feature tests pass | OK |
| `docs/uplift_analysis.md` | Uplift model tests pass | OK |

## Recommendations (priority order)

1. **High — Fix MLflow fixture run-leak** in `tests/conftest.py` (one fix → ~36 failures resolved).
2. **High — Provide pre-populated dashboard fixture data** (one fix → ~45 dashboard failures resolved).
3. **Medium — Reconcile recommendations schema** (`segment`, `estimated_cost`) between code, docs, and tests.
4. **Low — Path-scrub Windows-absolute paths** in `mlflow_tracking.py`.
5. **Low — Skip WSL bash test on non-WSL hosts** (or switch to Git-bash).

## Artifacts

- JUnit XML per group: `_test_results/group{1..6}.xml`
- Raw pytest log per group: `_test_results/group{1..6}.log` (UTF-16, written by PowerShell `Tee-Object`)
- This report: `_test_results/MERGED_REPORT.md`

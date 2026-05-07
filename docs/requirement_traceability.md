# Requirement Traceability

This note records where each item from `issue_final.md` was reflected in the repository.

| Requirement area | Reflected in |
| --- | --- |
| `results/` artifact completeness | `src/main.py` publishes training, uplift, CLV, segmentation, budget, A/B, cohort, monitoring, and checklist artifacts through `_save_result_and_artifact()` and `REQUIRED_PIPELINE_ARTIFACTS`. |
| Cohort outputs | `src/main.py::run_cohort`, `src/analysis/cohort_analysis.py`, `tests/test_cohort_analysis.py`, `tests/test_cohort_computations.py`. |
| 16-stage `run_all` pipeline | `src/main.py::run_all`, `src/pipeline/runner.py`, `tests/test_pipeline_runner.py`, `README.md`, `docs/README.md`, `docs/architecture.md`, `docs/usage.md`. |
| File feature store / Parquet engine | `requirements.txt`, `src/features/feature_engineering.py`, `src/main.py::run_features`, `tests/test_feature_engineering.py`. |
| Simulator small mode and treatment/churn guards | `config/simulator_config.yaml`, `docker-compose.yml`, `src/data/generator.py`, `src/data/orchestrator.py`, `tests/test_data_generator.py`. |
| SHAP and ML training artifacts | `src/main.py::run_train`, `src/models/shap_explainer.py`, `tests/test_shap_explainer.py`, `tests/test_main_cli.py`. |
| DL real sequence training and trainer selection | `src/main.py::run_train`, `src/models/dl_trainer.py`, `src/models/sequence_utils.py`, `tests/test_dl_trainer.py`, `tests/test_sequence_dataset.py`. |
| Uplift two-learner comparison and 4-quadrant segmentation | `src/main.py::run_uplift`, `src/models/uplift_model.py`, `src/dashboard/data_loader.py`, `tests/test_uplift.py`, `tests/test_uplift_model.py`, `tests/test_churn_uplift_segmentation_views.py`. |
| CLV validation and top-value outputs | `src/main.py::run_clv`, `src/models/clv_model.py`, `tests/test_clv.py`, `tests/test_clv_model.py`, `docs/models.md`. |
| Churn/uplift/CLV segmentation and priority score | `src/main.py::run_segment`, `src/dashboard/data_loader.py`, `src/features/segmentation.py`, `tests/test_segmentation.py`, `tests/test_clv_cohort_views.py`. |
| Budget optimization ROI and 50/100/200 what-if | `src/main.py::run_optimize`, `src/models/budget_optimizer.py`, `src/optimization/budget_optimizer.py`, `tests/test_budget_optimization.py`, `tests/test_budget_lp_solver.py`, `tests/test_lp_budget_optimizer.py`. |
| A/B testing real simulator data and detailed dashboard schema | `src/main.py::run_ab_test`, `src/models/ab_testing.py`, `src/dashboard/data_loader.py`, `tests/test_ab_testing.py`, `tests/test_dashboard.py`. |
| Dashboard loader path/schema alignment | `src/dashboard/data_loader.py`, `src/dashboard/app.py`, `src/dashboard/system_health_view.py`, `tests/test_dashboard.py`, `tests/test_streamlit_dashboard.py`, `tests/test_model_monitoring_view.py`. |
| Monitoring report and drift/performance history | `src/main.py::run_monitor`, `src/monitoring/drift_detection.py`, `src/monitoring/ks_drift.py`, `src/monitoring/monitoring_service.py`, `src/dashboard/data_loader.py`, `tests/test_drift_detection.py`, `tests/test_ks_drift.py`, `tests/test_model_monitoring_view.py`. |
| Documentation corrections | `README.md`, `docs/README.md`, `docs/architecture.md`, `docs/deployment.md`, `docs/models.md`, `docs/modules.md`, `docs/usage.md`, `docs/api.md`. |

Latest artifact evidence:

- `python src/main.py --mode all --small --quiet` completed after resetting `data/raw/pipeline_state.json`.
- `results/required_artifacts_checklist.json`: `25 / 25` satisfied, `missing: []`.
- `results/uplift_results.csv`: 4 quadrants present (`sure_thing`, `sleeping_dog`, `lost_cause`, `persuadable`) and includes `treatment_effect`.
- `results/segments_6plus.csv`: 6 operating segments present and includes `priority_score`.
- `results/budget_whatif.csv`: 50%, 100%, and 200% budget scenarios present.
- `results/ab_test_detailed.json`: multiple experiments with power, p-value, confidence interval, and Cohen's h.
- `results/monitoring_report.json`: PSI/KS summaries, threshold records, `overall_alert_level`, and `drifted_features`.

Latest verification evidence: `OMP_NUM_THREADS=1 LIGHTGBM_NUM_THREADS=1 /tmp/capstone-codex-py312/bin/python -m pytest -q` passed with `2503 passed, 85 warnings`.

## issue_final_v2 follow-up fixes

The fresh-context six-agent verification summarized in `issue_final_v2.md`
found five remaining blockers. The follow-up implementation was reflected here:

| issue_final_v2 blocker | Reflected in |
| --- | --- |
| Cohort M1/M3/M6/M12, churn-last-30 top-5, pre-churn event, journey funnel artifacts | `src/analysis/cohort_analysis.py` normalizes event schema and carries latest-observed retention into long-horizon milestone columns; `src/main.py::run_cohort` records exact/fallback milestone policy and writes `cohort_milestones.csv`, `churn_last30_sequences.json`, `pre_churn_events.csv`, and `journey_funnel.csv`. |
| Recommendation active actions for negative uplift / sleeping dogs | `src/models/recommendations.py` adds explicit `no_action` policy; `src/main.py::run_recommend` now builds recommendations from `segments_6plus.csv`, `uplift_results.csv`, and `clv_predictions.csv` with segment/uplift/CLV/churn/ROI context. |
| Stale small-run feature and pipeline resume risk | `src/main.py::_compute_features` validates cached feature rows/customer IDs against current input and current results directory; `src/main.py::run_all` resets checkpoint state when run context (`--small`, directories, simulation size, step order) changes. |
| Artifact checklist existence-only risk | `src/main.py::_write_artifact_checklist` now validates required artifact schemas/content and syncs existing `results/` files into `data/artifacts/`; the checklist now covers 25 artifacts including cohort milestones, churn sequences, journey outputs, and recommendations. |
| Dashboard monitoring/performance history fallback | `src/dashboard/data_loader.py` reads `model_performance_history.csv` for AUC/Precision/Recall histories and no longer hides an existing incomplete cohort matrix behind generated sample data. |

Latest focused verification after these fixes:

- `results/required_artifacts_checklist.json`: `25 / 25` satisfied, `missing: []`.
- `results/cohort_milestones.csv`: populated `M1`, `M3`, `M6`, `M12` columns.
- `results/churn_last30_sequences.json`: 5 sequence patterns.
- `results/recommendations.csv`: 5,000 rows with `recommendation_type`, `segment`, `uplift_score`, `clv`, `churn_probability`, `priority_score`, and `expected_roi`; active recommendations for no-action candidates: `0`.
- Targeted tests:
  - `tests/test_recommendations.py`: `33 passed`.
  - `tests/test_dashboard.py tests/test_streamlit_dashboard.py tests/test_pipeline_runner.py`: `319 passed`.
  - `tests/test_main_cli.py tests/test_cli_entrypoint.py tests/test_pipeline_runner.py`: `162 passed`.
  - `tests/test_budget_optimization.py tests/test_drift_detection.py tests/test_ks_drift.py tests/test_model_monitoring_view.py`: `172 passed`.

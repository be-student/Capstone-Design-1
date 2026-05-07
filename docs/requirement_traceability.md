# Requirement Traceability

This note records where each item from `issue_final.md` was reflected in the repository.

| Requirement area | Reflected in |
| --- | --- |
| `results/` artifact completeness | `src/main.py` publishes training, uplift, CLV, segmentation, budget, A/B, cohort, monitoring, and checklist artifacts through `_save_result_and_artifact()` and `REQUIRED_PIPELINE_ARTIFACTS`. |
| Cohort outputs | `src/main.py::run_cohort`, `src/analysis/cohort_analysis.py`, `tests/test_cohort_analysis.py`, `tests/test_cohort_computations.py`. |
| 16-stage `run_all` pipeline | `src/main.py::run_all`, `src/pipeline/runner.py`, `tests/test_pipeline_runner.py`, `README.md`, `docs/README.md`, `docs/architecture.md`, `docs/usage.md`. |
| File feature store / Parquet engine | `requirements.txt`, `src/features/feature_engineering.py`, `src/main.py::run_features`, `tests/test_feature_engineering.py`. |
| Simulator generation modes and treatment/churn guards | `config/simulator_config.yaml`, `docker-compose.yml`, `src/data/generator.py`, `src/data/orchestrator.py`, `tests/test_data_generator.py`. |
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

Current v4 artifact evidence:

- `python src/main.py --mode all` completed via resume and wrote `results/required_artifacts_checklist.json`.
- `results/required_artifacts_checklist.json`: `29 / 29` satisfied, `missing: []`, `full_submission_ready: true`.
- `data/raw/generation_summary.json`: full mode, 20,000 customers, treatment/control `10,000 / 10,000`, `churn_rate: 0.19995`, `group_size_check` passed, and `target_churn_check` passed.
- `results/cohort_analysis.json`: 4 cohorts, retention matrix shape `[4, 13]`, exact milestones `[1, 3, 6, 12]`, `fallback_milestones: []`, `journey_funnel_saved: true`, and `churn_sequence_observations: 407`.
- `results/journey_funnel.csv`: `Signup` count is 20,000.
- `results/segment_validation.json`: structured absence report is present, `absence_reason` is non-null, and `validation.valid: true` with reason `structured_absence_report_present`.
- `results/ab_test_detailed.json`: includes `required_sample_size_per_group`, `required_total_sample_size`, `observed_power`, `design_power`, `is_underpowered`, `power_status`, and `statistically_significant`.

Current v4 verification evidence: integration targeted pytest passed with `1050 passed, 1 skipped, 6 warnings`.

## issue_final_v2 follow-up fixes

The fresh-context six-agent verification summarized in `issue_final_v2.md`
found five remaining blockers. The follow-up implementation was reflected here:

| issue_final_v2 blocker | Reflected in |
| --- | --- |
| Cohort M1/M3/M6/M12, churn-last-30 top-5, pre-churn event, journey funnel artifacts | `src/analysis/cohort_analysis.py` normalizes event schema and computes exact M1/M3/M6/M12 milestones for the current full run; `src/main.py::run_cohort` records milestone policy and writes `cohort_milestones.csv`, `churn_last30_sequences.json`, `pre_churn_events.csv`, and `journey_funnel.csv`. |
| Recommendation active actions for negative uplift / sleeping dogs | `src/models/recommendations.py` adds explicit `no_action` policy; `src/main.py::run_recommend` now builds recommendations from `segments_6plus.csv`, `uplift_results.csv`, and `clv_predictions.csv` with segment/uplift/CLV/churn/ROI context. |
| Stale cached feature and pipeline resume risk | `src/main.py::_compute_features` validates cached feature rows/customer IDs against current input and current results directory; `src/main.py::run_all` validates checkpoint state against run context, directories, simulation size, and step order. |
| Artifact checklist existence-only risk | `src/main.py::_write_artifact_checklist` now validates required artifact schemas/content and syncs existing `results/` files into `data/artifacts/`; the checklist now covers 29 artifacts including cohort milestones, churn sequences, journey outputs, recommendations, and generation summary validation. |
| Dashboard monitoring/performance history loading | `src/dashboard/data_loader.py` reads `model_performance_history.csv` for AUC/Precision/Recall histories and no longer hides an existing incomplete cohort matrix behind generated sample data. |

Current focused verification after these fixes:

- `results/required_artifacts_checklist.json`: `29 / 29` satisfied, `missing: []`, `full_submission_ready: true`.
- `data/raw/generation_summary.json`: full mode with 20,000 customers, treatment/control `10,000 / 10,000`, and passing group-size/churn-target validation.
- `results/cohort_analysis.json`: 4 cohorts, retention matrix shape `[4, 13]`, exact milestones `[1, 3, 6, 12]`, zero milestone substitutions, and 407 churn-sequence observations.
- `results/journey_funnel.csv`: `Signup` count is 20,000.
- `results/segment_validation.json`: structured absence report present; `absence_reason` is populated; validation is true with reason `structured_absence_report_present`.
- `results/ab_test_detailed.json`: power-analysis and significance fields are present for detailed A/B reporting.
- Integration targeted pytest: `1050 passed, 1 skipped, 6 warnings`.

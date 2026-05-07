# Requirement Traceability

This note records where the `require.md` submission guardrails and the v4
blocker-closure evidence are reflected in the repository. `issue_final_v4.md`
is already PASS, so this file is the current traceability index for
requirement-level verification.

| Requirement area | Requirement anchor | Reflected in |
| --- | --- | --- |
| Docker one-command execution and dashboard | §2, §7, §9 | `docker-compose.yml`, `README.md`, `docs/deployment.md`, `docs/architecture.md`, `tests/test_integration.py`, `tests/test_docker_setup.py`. |
| CLI modes `train`, `uplift`, `optimize`, and `all` | §2, §9 | `src/main.py::MODES`, `src/main.py::build_parser`, `src/main.py::run_all`, `src/pipeline/runner.py`, `tests/test_main_cli.py`, `tests/test_cli_entrypoint.py`, `tests/test_pipeline_runner.py`. |
| Required artifact completeness and mirror validation | §2 checklist, §7, final output checklist | `src/main.py` publishes required artifacts through `_save_result_and_artifact()` and validates them with `_write_artifact_checklist()` / `REQUIRED_PIPELINE_ARTIFACTS`; dashboard mirrors are checked by SHA-256. |
| Simulator full-mode treatment/churn guards | §5.1, §7, §9 | `config/simulator_config.yaml`, `src/data/generator.py`, `src/data/orchestrator.py`, `src/pipeline/artifact_validation.py`, `tests/test_data_generator.py`. |
| Cohort and journey outputs | §5.2, §9 | `src/main.py::run_cohort`, `src/analysis/cohort_analysis.py`, `src/pipeline/artifact_validation.py`, `tests/test_cohort_analysis.py`, `tests/test_cohort_computations.py`, `tests/test_pipeline_runner.py`. |
| Feature store, feature dictionary, and business meaning | §5.3, §5.13, G3 | `src/features/feature_engineering.py`, `src/main.py::run_features`, `docs/feature_dictionary.md`, `tests/test_feature_engineering.py`. |
| ML/DL training, SHAP, and model report | §5.4, §5.5, §5.13, §9 | `src/main.py::run_train`, `src/models/churn_model.py`, `src/models/dl_trainer.py`, `src/models/shap_explainer.py`, `docs/model_report.md`, `docs/models.md`, `tests/test_churn_model.py`, `tests/test_dl_trainer.py`, `tests/test_shap_explainer.py`. |
| Uplift modeling and four-quadrant segmentation | §5.6, §5.13, §9 | `src/main.py::run_uplift`, `src/models/uplift_model.py`, `docs/uplift_analysis.md`, `tests/test_uplift.py`, `tests/test_uplift_model.py`. |
| CLV prediction and validation | §5.7, §9 | `src/main.py::run_clv`, `src/models/clv_model.py`, `docs/models.md`, `tests/test_clv.py`, `tests/test_clv_model.py`. |
| Six-segment priority scoring | §5.8, §9 | `src/main.py::run_segment`, `src/features/segmentation.py`, `results/segments_6plus.csv`, `tests/test_segmentation.py`. |
| Budget optimization and retention strategy | §5.9, §9 | `src/main.py::run_optimize`, `src/models/budget_optimizer.py`, `src/optimization/budget_optimizer.py`, `docs/retention_strategy.md`, `tests/test_budget_optimization.py`, `tests/test_budget_lp_solver.py`, `tests/test_lp_budget_optimizer.py`. |
| A/B test power, p-value, and confidence interval evidence | §5.10, §7, §9 | `src/main.py::run_ab_test`, `src/models/ab_testing.py`, `docs/ab_test_report.md`, `tests/test_ab_testing.py`, `tests/test_ab_statistical_methods.py`, `tests/test_statistical_testing.py`. |
| Dashboard loader path/schema alignment | §5.11, G7 | `src/dashboard/data_loader.py`, `src/dashboard/app.py`, `src/dashboard/system_health_view.py`, `tests/test_dashboard.py`, `tests/test_streamlit_dashboard.py`, `tests/test_model_monitoring_view.py`. |
| Monitoring report and drift/performance history | §5.12, §9 | `src/main.py::run_monitor`, `src/monitoring/drift_detection.py`, `src/monitoring/ks_drift.py`, `src/monitoring/monitoring_service.py`, `tests/test_drift_detection.py`, `tests/test_ks_drift.py`. |
| Documentation, architecture diagram, docstrings, and module separation | §5.13, §7, G1-G7 | `README.md`, `docs/README.md`, `docs/architecture.md`, `docs/deployment.md`, `docs/models.md`, `docs/modules.md`, `docs/usage.md`, `docs/api.md`, module-level tests under `tests/`. |

Current v4 artifact evidence:

- `python src/main.py --mode all` completed via resume and wrote `results/required_artifacts_checklist.json`.
- `results/required_artifacts_checklist.json`: `29 / 29` satisfied, `missing: []`, `full_submission_ready: true`.
- `data/raw/generation_summary.json`: full mode, 20,000 customers, treatment/control `10,000 / 10,000`, `churn_rate: 0.19995`, `group_size_check` passed, and `target_churn_check` passed.
- `results/cohort_analysis.json`: 4 cohorts, retention matrix shape `[4, 13]`, exact milestones `[1, 3, 6, 12]`, `fallback_milestones: []`, `journey_funnel_saved: true`, and `churn_sequence_observations: 407`.
- `results/journey_funnel.csv`: `Signup` count is 20,000.
- `results/segment_validation.json`: structured absence report is present, `absence_reason` is non-null, and `validation.valid: true` with reason `structured_absence_report_present`.
- `results/ab_test_detailed.json`: includes `required_sample_size_per_group`, `required_total_sample_size`, `observed_power`, `design_power`, `is_underpowered`, `power_status`, and `statistically_significant`.

Current v4 verification evidence: targeted pytest suite passed with `1339 passed, 1 skipped, 6 warnings`.

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
- Integration targeted pytest: `1339 passed, 1 skipped, 6 warnings`.

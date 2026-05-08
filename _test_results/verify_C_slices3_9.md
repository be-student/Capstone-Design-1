# Verifier C — Slices #3 (Cohort & Journey) & #9 (Customer Segmentation)

| Slice | Verdict |
| --- | --- |
| #3 Cohort & Customer Journey | **PASS** (with environmental caveat) |
| #9 6+ Customer Segments | **PASS** |

## Slice #3 — Cohort & Customer Journey: PASS (with caveat)

- `src/main.py::run_cohort` (lines 1977-2171) writes all five required artifacts (`cohort_milestones.csv`, `churn_last30_sequences.json`, `pre_churn_events.csv`, `journey_funnel.csv`, `cohort_churn_rate_differences.png`) and validates them through `REQUIRED_PIPELINE_ARTIFACTS` (lines 85-90) and `_artifact_content_valid` (lines 413-528, 580). Milestone policy M1/M3/M6/M12 + 5-stage funnel with Signup are enforced.
- The two Group-3 failures (`tests/test_cohort_computations.py:589` and `:608`, `TestCohortDashboardIntegration`) both call `DashboardDataLoader.load_cohort_data()`. `src/dashboard/data_loader.py:938-1024` returns an empty DataFrame when neither `results/cohort_data.csv` nor `data/raw/events.*` exist. Glob confirms **all three paths are absent** in this checkout. The empty frame propagates through `assign_cohorts → compute_retention_matrix` and triggers `assert 0 > 0`. This is RCA #2 in MERGED_REPORT (env/data, not code).
- Group 3 = 379/381 PASS. `tests/test_cohort_analysis.py` (pure cohort logic, no loader) is fully green; only the two integration tests bound to the loader fail.

## Slice #9 — 6+ Customer Segments: PASS

- `priority_score = uplift_score * clv` implemented literally at `src/main.py:2236` and `src/features/segmentation.py:560`; the validator at `src/main.py:458-468` requires the column in `segments_6plus.csv`. `run_segment` (lines 2185-2280) yields ≥9 distinct segment labels (`{high,mid,low}_value_{persuadable,lost_cause,sure_thing}` + `sleeping_dog`); RFM `CustomerSegmenter` retains 8 named segments (`src/features/segmentation.py:33-180`).
- `tests/test_segmentation.py` (Group 1, 167 tests) — **0 failures**. Segment summary, visualization, schema validator all wired (`src/main.py:2280-2321`).
- The 3 `test_clv_cohort_views` failures in Group 5 (`test_clv_data_nonempty`, `test_clv_data_has_multiple_segments`, `test_retention_matrix_from_loaded_data`) are the same loader-empty cascade — `results/clv_predictions.csv`, `results/segments_6plus.csv`, `results/cohort_data.csv` and `data/raw/events.*` are all absent in this checkout (verified by Glob). Not a segmentation defect.

## Skeptical caveat on v8's PASS

v8's verification commands ran on a freshly populated pipeline state (e.g. `pytest -k "cohort..." → 10 passed`). The current MERGED_REPORT was produced **without that pre-run**, so the dashboard-integration tests fail because their on-disk fixtures don't exist. v8's PASS is therefore **defensible but environment-dependent**: it requires the simulator/cohort/CLV/segment pipeline to be executed before the dashboard-coupled integration tests run. From a clean checkout, those 5 fails (2 cohort + 3 CLV/cohort-views) are inevitable until artifacts are produced.

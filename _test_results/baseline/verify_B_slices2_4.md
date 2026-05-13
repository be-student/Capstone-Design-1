# Verifier B — Slices #2 (Simulator) & #4 (Feature Engineering)

## Slice #2 — Customer Behavior Simulator: PASS

- **Population & split enforced in code:** `config/simulator_config.yaml:14` sets `num_customers: 20000`; `:43` sets `treatment_ratio: 0.50`, `min_group_size: 10000`. `src/data/orchestrator.py:213-224` (`_validate_group_sizes`) and `src/pipeline/artifact_validation.py:47-79` re-validate `treatment_count`/`control_count` and `target_churn_check`.
- **Six personas with proportions summing to 1.00:** `config/simulator_config.yaml:241-426` declares `vip_loyal(0.10) + regular_loyal(0.25) + bargain_hunter(0.20) + explorer(0.15) + dormant(0.15) + new_customer(0.15) = 1.00` (matches the c70d7cd fix). Each carries a `marketing_response` block. Raw events schema includes `session_duration, marketing_channel, marketing_response` at `src/data/generator.py:44-46` and is populated at L278-302 and L497-615.
- **Tests pass:** Group 1 Data & Features = 167/167 PASS in `_test_results/MERGED_REPORT.md`. Group 6's 43 failures contain zero `test_integration::test_six_personas_in_generated_data` or `test_treatment_control_split` entries — those tests pass post-UTF8.

Caveat: `data/raw/generation_summary.json` is absent on this snapshot, so the `0.19995` churn rate is taken from `docs/requirement_traceability.md:30` and `issue_final_v8.md:54`, not re-observed in this verification run.

## Slice #4 — Feature Engineering: PASS

- **33 feature columns across 7 groups, computed by `compute_all_features` at `src/features/feature_engineering.py:49`:** RFM(5)+behavioral(7)+anomaly(3)+session(5)+sequence(4)+time(6)+journey&tenure(3)=33. Group entrypoints: `compute_rfm` L119, `compute_behavioral_changes` L186, `compute_purchase_cycle_anomaly` L345, `compute_session_quality` L414, `compute_sequence_features` L516, `compute_time_features` L612, `compute_journey_features` L698.
- **Null/inf handling at `src/features/feature_engineering.py:917-939` (`_sanitize_feature_matrix`):** runs at end of `compute_all_features` (L113), executes `result[col].replace([np.inf, -np.inf], np.nan).fillna(0)` (L930) plus 1st/99th-percentile winsorisation. `_safe_ratio` (L894-915) guards per-row divisions; RFM has explicit guards at L160-163.
- **Tests pass:** Group 1 = 167/167 PASS, covering `test_feature_engineering` and `test_segmentation`. No feature-engineering test appears in any other group's failure list. `docs/feature_dictionary.md` documents each group's computation, range, business meaning, and rationale.

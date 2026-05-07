# issue_final_v4.md

## 목적

이번 문서는 `require.md`를 필수 가드레일로 두고, `issue_final_v3.md`의 남은 blocker를 닫는 v4 사이클의 작업, 검증, 최종 판정을 기록한다.

시작 조건:
- 기준 문서: `issue_final_v3.md`
- 생성 시각: 2026-05-07T16:21:36Z
- 운영 원칙: 메인 컨텍스트는 오케스트레이션만 담당하고, 구현 수정은 fresh-context executor에게 위임한다.

## v3 기준 남은 blocker

1. full-mode evidence 부재 및 checklist의 full/small 검증 부족
2. cohort/journey stale artifact와 fail-fast 부족
3. `results/`와 `data/artifacts/` mirror 신뢰성 부족
4. dashboard all-customer churn predictions, monitoring history wiring, sample fallback 문제
5. high-value actionable segmentation evidence 부족
6. A/B detailed sample-size/power schema 부족
7. local SHAP, CLV actual-vs-predicted validation, uplift best learner selection 부족
8. repository hygiene: `.audit/`, tracked generated model/pycache residue

## 작업 요약

| Lane | 담당 agent | 주요 변경 |
| --- | --- | --- |
| Pipeline/artifacts/cohort | executor | full/small `generation_summary.json` 검증, strict mirror hash 검증, cohort/journey exact milestone/population/churn-sequence validation, `run_all` full_submission_ready gate, failed cohort checklist invalidation |
| Dashboard/monitoring | executor | required dashboard artifact missing/invalid 시 sample fallback 대신 visible empty/error state, all-customer churn coverage warning, real model performance history wiring |
| ML/SHAP/uplift/CLV | executor | local SHAP artifact export, uplift `auto` best-AUUC learner selection, CLV future-window actual-vs-predicted validation, native-crash-safe churn/SHAP test coverage |
| Segmentation/A-B | executor | high-value actionable structured absence report, strict segment validation, A/B detailed power/sample-size schema with `power_status` and `statistically_significant` |
| Test isolation/hygiene | executor | CLI/pipeline tests isolate temp data/results/artifact dirs; regression guard verifies tests do not mutate default full-run artifacts |
| Traceability docs | writer | `docs/requirement_traceability.md` 최신 full-mode evidence로 갱신, stale small-mode/test-count claims 제거 |

## 현재 산출물 evidence

- `data/raw/generation_summary.json`: full mode, 20,000 customers, treatment/control 10,000/10,000, churn rate 0.19995, group size 및 target churn validation passed.
- `results/required_artifacts_checklist.json`: 29/29 satisfied, `missing: []`, `full_submission_ready: true`.
- `results/`와 `data/artifacts/`: required checklist 대상 artifact mirror hash 일치.
- `results/cohort_analysis.json`: 4 cohorts, retention matrix `[4, 13]`, exact milestones `[1, 3, 6, 12]`, fallback milestones `[]`, churn sequence observations 407.
- `results/journey_funnel.csv`: `Signup` count 20,000.
- `results/churn_predictions.csv`: 20,000 rows / 20,000 unique customers.
- `results/uplift_learner_comparison.csv`: `s_learner` AUUC가 `t_learner`보다 높고, `results/uplift_results.csv`는 selected learner를 `s_learner`로 기록.
- `results/clv_validation.json`: target `future_revenue_12m_actual`, future label window 91 days, `results/clv_actual_vs_predicted.csv` 4,000 holdout rows.
- `results/segment_validation.json`: high-value actionable count 0에 대해 structured absence report present, `validation.valid: true`, reason `structured_absence_report_present`.
- `results/ab_test_detailed.json`: `required_sample_size_per_group`, `required_total_sample_size`, `observed_power`, `design_power`, `is_underpowered`, `power_status`, `statistically_significant` 포함.

## 검증 evidence

- `uv run --python /opt/homebrew/bin/python3.12 --with pytest --with-requirements requirements.txt pytest -q tests/test_pipeline_runner.py tests/test_cohort_analysis.py tests/test_cohort_computations.py tests/test_dashboard.py tests/test_streamlit_dashboard.py tests/test_model_monitoring_view.py tests/test_churn_analytics_views.py tests/test_churn_uplift_segmentation_views.py tests/test_clv_cohort_views.py tests/test_ab_testing.py tests/test_ab_statistical_methods.py tests/test_statistical_testing.py tests/test_segmentation.py tests/test_churn_model.py tests/test_shap_explainer.py tests/test_uplift.py tests/test_uplift_model.py tests/test_clv.py tests/test_clv_model.py tests/test_dl_trainer.py tests/test_main_cli.py tests/test_cli_entrypoint.py tests/test_docker_setup.py`
  - 결과: 1339 passed, 1 skipped, 6 warnings.
- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt python src/main.py --mode all --quiet`
  - 결과: success.
- CLI/pipeline test artifact isolation check:
  - 주요 full-run artifact hash 전후 변경 수: 0.
  - `validate_cohort_artifacts(Path("results"), Path("data/raw"))`: `valid: true`, `errors: []`, `expected_customer_count: 20000`.
- Static checks:
  - `git diff --check`: passed.
  - 변경 Python 파일 `py_compile`: passed.

## Fresh-context verifier 결과

최종 6-agent verifier rerun 결과는 모두 PASS다.

| 관점 | 판정 | 핵심 결론 |
| --- | --- | --- |
| CLI / Docker / pipeline / artifact completeness | PASS | full-mode generation, strict cohort/journey validation, mirror trust, `run_all` full_submission_ready gate, stale checklist invalidation, CLI test isolation, Docker config/CLI checks passed. |
| Simulator / cohort / journey / feature engineering | PASS | 20,000-customer full simulator evidence, exact M1/M3/M6/M12 cohort milestones, journey signup 20,000, churn sequence observations 407, feature store and 33 documented features verified. |
| ML / DL / SHAP / uplift / CLV | PASS | local SHAP artifacts, non-crashing churn/DL tests, uplift best learner selection, CLV temporal actual-vs-predicted validation, and ML/DL/SHAP/uplift/CLV mirrors verified. |
| Segmentation / budget / recommendations / A-B | PASS | 6 segments, priority score equality, structured high-value absence report, no-action budget/recommendation constraints, A/B power/sample-size schema, and mirror equality verified. |
| Dashboard / loader / monitoring | PASS | required artifact loaders expose missing/invalid states, model metrics empty-state, all-customer churn coverage, CLV zero-row preservation, monitoring report and real performance history wiring verified. |
| Docs / traceability / hygiene / commit readiness | PASS | stale small-mode/latest evidence removed, no tracked generated residue, ignored local artifacts confirmed, `issue_final_v4.md` current evidence recorded. |

## 종합 판정

최종 판정: PASS.

`issue_final_v3.md`의 남은 blocker는 v4 사이클에서 모두 닫혔다. 다음 사이클은 더 이상 `issue_final_v3.md`만 기준으로 보지 않고, 사용자 지침대로 `require.md` 전체를 6개 fresh-context agent로 다시 분해 분석하여 잔여 gap을 찾는 require-first 검증/수정 루프로 전환한다.

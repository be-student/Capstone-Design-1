# Docker Full Pipeline 재검증 체크리스트 초안

## 목적

Docker full pipeline을 새로 실행한 뒤 `require.md` 전체를 빠르게 재검증하기 위한 자동/반자동 체크리스트다.
코드/compose/Dockerfile 수정 없이, 현재 산출물 구조와 `issue_final_v8.md`의 12조각 PASS 기준을 기준으로 한다.

## 현재 기준 요약

- 최신 최종 판정 문서: `issue_final_v8.md`
- 최신 traceability index: `docs/requirement_traceability.md`
- 필수 요구사항 원문: `require.md`
- 현재 checklist 산출물: `results/required_artifacts_checklist.json`
- 현재 full-mode generation evidence: `data/raw/generation_summary.json`
- 현재 Docker/CLI 기준: `docker-compose.yml`, `Dockerfile.pipeline`, `scripts/pipeline_entrypoint.sh`

기존 issue 흐름:

- `issue_1.md` ~ `issue_4.md`: 초기 감사에서 산출물 부재, dead-code, CLI/pipeline 연결, segmentation/uplift/monitoring gap 확인.
- `issue_final.md` ~ `issue_final_v3.md`: small/stale artifact, cohort/journey, checklist mirror, dashboard fallback, A/B/SHAP/CLV evidence blocker 정리.
- `issue_final_v4.md` ~ `issue_final_v6.md`: full-mode evidence, strict artifact validation, dashboard loader, class imbalance, uplift direction, Redis Docker env 등 닫고 PASS.
- `issue_final_v7.md`: 12조각 재검증 중 5개 FAIL. 주요 blocker는 raw event schema, cohort churn-rate visualization, journey timing, A/B balance evidence, monitoring performance alert, GitHub Flow docs.
- `issue_final_v8.md`: v7 blocker를 닫고 12조각 모두 PASS. 이 문서의 분해 방식을 Docker 재검증 기준으로 유지한다.

## Docker full pipeline 재실행 절차

사전 확인:

```bash
git status --short
docker compose config --quiet
```

full submission evidence 재생성:

```bash
PIPELINE_MODE=all SMALL=false VERBOSE=false docker compose up --build pipeline
```

대시보드까지 확인할 때:

```bash
SKIP_PIPELINE=true docker compose up --build dashboard
curl -f http://localhost:8501/_stcore/health
```

파이프라인 완료 후 공통 자동 검증:

```bash
jq '.generation_summary_validation, {required_count, satisfied_count, missing, full_submission_ready}' results/required_artifacts_checklist.json
jq '.artifacts | map(select(.satisfied != true or .mirror_hash_match != true)) | length' results/required_artifacts_checklist.json
jq '{generation_mode, num_customers, treatment_count, control_count, churn_rate, num_events, validation}' data/raw/generation_summary.json
cmp results/required_artifacts_checklist.json data/artifacts/required_artifacts_checklist.json
git diff --check
```

대표 회귀 테스트:

```bash
uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest -p no:cacheprovider tests/test_cli_entrypoint.py tests/test_main_cli.py tests/test_pipeline_runner.py tests/test_docker_setup.py
uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest tests/test_pipeline_runner.py -k "artifact or checklist or cohort or generation_summary"
uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest tests/test_generation_summary_schema.py tests/test_data_generator.py tests/test_orchestrator.py
uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest tests/test_dashboard.py tests/test_dashboard_helpers.py tests/test_streamlit_dashboard.py tests/test_model_monitoring_view.py
```

전체 신뢰도 확인이 필요하면 마지막에 실행:

```bash
uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest -q
```

## 12조각 재검증 기준

| 조각 | 범위 | 자동/반자동 PASS 기준 | 핵심 증거 |
| --- | --- | --- | --- |
| 1 | 최종 실행/산출물 계약 | Compose config 통과, CLI help/mode 존재, checklist가 `required_count == satisfied_count`, `missing == []`, `full_submission_ready == true`, mirror mismatch 0 | `docker compose config --quiet`; `results/required_artifacts_checklist.json`; `data/artifacts/required_artifacts_checklist.json`; CLI/pipeline tests |
| 2 | 고객 행동 시뮬레이터 | full mode 20,000명 이상, treatment/control 각 10,000명 이상, churn 15~25%, 8개 event type, raw events에 `session_duration`, `marketing_channel`, `marketing_response` 존재, persona별 response evidence 존재 | `data/raw/generation_summary.json`; `data/raw/customers.csv`; `data/raw/events.csv`; `tests/test_generation_summary_schema.py`; `tests/test_data_generator.py` |
| 3 | 코호트 및 고객 여정 | M1/M3/M6/M12 exact milestone, fallback milestone 없음, retention matrix/curve/heatmap 존재, churn-rate-difference PNG 존재 및 mirror 동일, last30 top-5 sequence, pre-churn events, journey timing/dropoff evidence 존재 | `results/cohort_analysis.json`; `results/cohort_milestones.csv`; `results/cohort_retention_matrix.csv`; `results/cohort_churn_rate_differences.png`; `results/churn_last30_sequences.json`; `results/pre_churn_events.csv`; `results/journey_funnel.csv` |
| 4 | 피처 엔지니어링 | 30개 이상 feature, RFM/행동 변화율/구매 주기 이상/세션 품질/시퀀스/시간대/여정 stage feature 존재, row 수가 고객 수와 일치, null/inf/outlier sanitation evidence, feature store/mirror 존재 | `results/features.csv`; `data/feature_store/features.csv` 또는 parquet; `docs/feature_dictionary.md`; `tests/test_feature_engineering.py` |
| 5 | ML 이탈 모델 | XGBoost/LightGBM 등 2개 tree model 비교, class imbalance 처리, 5-fold CV/tuning, AUC >= 0.78, SHAP summary/local, top-10 feature importance, threshold analysis 존재 | `results/model_metrics.json`; `results/feature_importance.csv`; `results/shap_summary.png`; `results/shap_local_explanations.csv`; `results/threshold_analysis.json`; `docs/model_report.md`; `tests/test_churn_model.py`; `tests/test_shap_explainer.py` |
| 6 | DL 이탈 모델 | LSTM 또는 Transformer sequence model, padding/embedding, early stopping, ML/DL/ensemble 동일 test-set 비교, DL model/log/versioned artifact 존재 | `results/model_metrics.json`; `results/dl_training_log.json`; `models/dl_churn_model.pt`; `models/dl_churn_model_v1.pt`; `models/model_artifacts_manifest.json`; `tests/test_sequence_dataset.py`; `tests/test_dl_trainer.py` |
| 7 | Uplift Modeling | Treatment/Control 데이터 사용, T/S learner 등 2개 비교, 고객별 CATE/uplift score 20,000명, 4분면 segment, Qini curve, Persuadables analysis 문서/산출물 존재 | `results/uplift_results.csv`; `results/uplift_learner_comparison.csv`; `results/qini_curve.png`; `docs/uplift_analysis.md`; `tests/test_uplift.py`; `tests/test_uplift_model.py` |
| 8 | CLV 예측 | 12개월 기준 고객별 CLV, row 수 고객 수와 일치, high-value top 20%, actual-vs-predicted validation이 proxy가 아닌 future revenue target, top-N/distribution report 존재 | `results/clv_predictions.csv`; `results/clv_actual_vs_predicted.csv`; `results/clv_validation.json`; `results/clv_top_customers.csv`; `results/clv_distribution.json`; `tests/test_clv.py`; `tests/test_clv_model.py` |
| 9 | 고객 세그먼테이션 | 최소 6 segments, segment summary count/ratio/avg CLV/churn/uplift, `priority_score == uplift_score * clv`, visualization 존재, high-value actionable이 없으면 structured absence report 존재 | `results/segments_6plus.csv`; `results/segment_summary.csv`; `results/segment_validation.json`; `results/segments_6plus.png`; `tests/test_segmentation.py` |
| 10 | 리텐션 전략/예산/A-B | 세그먼트별 전략/비용/효과 문서, LP 또는 greedy budget optimization, 50/100/200% what-if, ROI, A/B p-value/95% CI/power/sample size, Treatment/Control balance/SMD evidence 존재 | `docs/retention_strategy.md`; `results/budget_optimization.csv`; `results/budget_optimization_summary.json`; `results/budget_whatif.csv`; `results/ab_test_detailed.json`; `results/ab_test_balance_check.csv`; `docs/ab_test_report.md` |
| 11 | 대시보드/모니터링 | Streamlit 8501 health, churn/cohort/uplift/CLV/budget/A-B/priority/6-segment/manual refresh 화면, required artifact fallback 없음, PSI/KS drift와 AUC/Precision/Recall performance degradation alert 존재 | `http://localhost:8501`; `results/monitoring_report.json`; `results/model_performance_history.csv`; dashboard tests; `tests/test_monitoring_service.py`; `tests/test_ks_drift.py`; `tests/test_drift_detection.py` |
| 12 | 제약/문서/품질/보너스 | README architecture diagram, docs deliverables, GitHub Flow docs, make_dataset, model versioning, MLflow evidence, dashboard logic split, tracked local absolute path 없음, git diff/check clean | `README.md`; `docs/development_workflow.md`; `docs/requirement_traceability.md`; `src/data/make_dataset.py`; `models/model_artifacts_manifest.json`; `tests/test_make_dataset.py`; `tests/test_mlflow_tracking.py`; `git diff --check` |

## 빠른 산출물 스키마 체크

아래 명령은 full pipeline 완료 후 PASS/FAIL을 빠르게 눈으로 확인하기 위한 최소 세트다.

```bash
# 1. checklist and mirror
jq '{required_count, satisfied_count, missing, full_submission_ready, generation_summary_validation}' results/required_artifacts_checklist.json
jq '.artifacts | map(select(.satisfied != true or .mirror_hash_match != true))' results/required_artifacts_checklist.json

# 2. simulator
jq '{generation_mode, num_customers, treatment_count, control_count, churn_rate, event_type_distribution, validation}' data/raw/generation_summary.json
head -n 1 data/raw/events.csv

# 3. cohort/journey
jq '{retention_matrix_shape, exact_milestones, fallback_milestones, churn_sequences_saved, pre_churn_events_saved, journey_funnel_saved, churn_sequence_observations}' results/cohort_analysis.json
test -s results/cohort_churn_rate_differences.png && cmp results/cohort_churn_rate_differences.png data/artifacts/cohort_churn_rate_differences.png
head -n 6 results/journey_funnel.csv

# 4. row counts and priority equality
wc -l data/raw/customers.csv results/features.csv results/churn_predictions.csv results/uplift_results.csv results/clv_predictions.csv results/segments_6plus.csv
awk -F, 'NR==1{for(i=1;i<=NF;i++) h[$i]=i} NR>1{d=$h["priority_score"]-($h["uplift_score"]*$h["clv"]); if(d<0)d=-d; if(d>max)max=d} END{print "max_priority_delta", max+0}' results/segments_6plus.csv

# 5. model/uplift/clv/budget/A-B/monitoring summaries
jq '{ml_model, dl_model, ensemble, top_features: (.top_features[0:10])}' results/model_metrics.json
head -n 5 results/uplift_learner_comparison.csv
jq '.' results/clv_validation.json
jq '{total_budget, allocated, expected_revenue_saved_krw, roi, what_if_scenarios}' results/budget_optimization_summary.json
jq '.experiments[] | {name, treatment_size, control_size, p_value, confidence_interval, observed_power, design_power, required_sample_size_per_group, statistically_significant, power_status}' results/ab_test_detailed.json
jq '{performance_degradation, performance_alerts, psi_report, ks_report}' results/monitoring_report.json
```

## 수동 확인 포인트

- Docker full rerun 직후 `generation_mode`가 `small`이면 제출 PASS로 보지 않는다.
- `results/`와 `data/artifacts/`의 mirror hash mismatch가 하나라도 있으면 dashboard evidence는 FAIL로 본다.
- `results/required_artifacts_checklist.json`이 PASS여도, v7 blocker였던 아래 항목은 별도로 확인한다.
  - raw events header에 `session_duration`, `marketing_channel`, `marketing_response` 존재.
  - `cohort_churn_rate_differences.png` 존재 및 mirror 동일.
  - `journey_funnel.csv`에 stage timing/dropoff timing 컬럼 존재.
  - `ab_test_balance_check.csv/json`의 pre-treatment confounder SMD PASS.
  - `monitoring_report.json`에 performance degradation alert 필드 존재.
  - `docs/development_workflow.md`에 GitHub Flow/branch/PR/CI/merge 정책 존재.
- 현재 작업트리에 다른 작업자의 수정이 있으면 검증 전후 `git status --short`를 기록하고, unrelated change는 되돌리지 않는다.

## 최종 판정 문구 템플릿

PASS일 때:

```text
Docker full pipeline 재실행 후 `require.md`를 v8 기준 12조각으로 재검증했다.
`results/required_artifacts_checklist.json`은 N/N satisfied, missing=[], full_submission_ready=true이고,
full-mode generation은 고객 20,000명, treatment/control 10,000/10,000, churn 15~25% 범위를 만족했다.
12조각 모두 PASS이며 남은 blocker는 없다.
```

FAIL일 때:

```text
Docker full pipeline 재실행 후 12조각 중 X개가 FAIL이다.
제출 차단 blocker는 [조각 번호/범위]이며, 실패 증거는 [명령 출력/파일/필드]이다.
기존 작업자 변경은 되돌리지 않았고, 후속 수정이 필요한 파일 범위는 [파일 목록]이다.
```

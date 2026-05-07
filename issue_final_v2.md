# issue_final_v2.md

## 목적

`issue_final.md`를 마무리하기 위한 현재 구현 상태를 `require.md` 필수 가드레일 기준으로 다시 검증했다. 검증은 기존 대화 컨텍스트를 넘기지 않은 6개 agent가 서로 다른 관점에서 read-only로 수행했다. 사용자의 지시에 따라 full test suite는 새로 실행하지 않았고, 코드/문서/현재 산출물/가벼운 정적 확인 중심으로 판단했다.

## 6개 agent 검증 결과

| 관점 | Agent | 판정 | 핵심 결론 |
| --- | --- | --- | --- |
| CLI / Docker / pipeline / artifact completeness | Poincare | FAIL | 필수 산출물 파일은 대부분 존재하지만, 현재 evidence가 small run이며 resume/cache가 stale small artifact를 재사용할 수 있다. Cohort artifact도 미완성이다. |
| Simulator / cohort / journey / feature engineering | Hooke | FAIL | simulator config와 feature engineering은 상당 부분 충족하지만, M1/M3/M6/M12 cohort milestone과 churn-last-30 sequence top-5가 실패한다. |
| ML / DL / SHAP / uplift / CLV | Ramanujan | PASS | ML/DL/ensemble, SHAP, T/S learner 비교, 4분면 uplift, CLV validation/top-20%, 6-segment artifact는 scoped 기준 통과했다. |
| Segmentation / budget / recommendation / A/B | Schrodinger | FAIL | segmentation/budget/A-B 핵심 산출물은 개선됐지만, recommendation이 negative uplift/sleeping dog 고객에게도 active action을 준다. |
| Dashboard / loader / monitoring | Bernoulli | FAIL | dashboard와 monitoring 구조는 개선됐지만, cohort view가 valid current output이 아니라 fallback/sample로 가려질 수 있고 performance history wiring이 미완성이다. |
| Docs / traceability / hygiene / commit readiness | Dewey | FAIL | 문서와 코드 반영은 진전됐지만, 현재 evidence가 small run이고 tracked generated files/pycache 등 repository hygiene 문제가 남아 있다. |

## 종합 판정

현재 구현은 `issue_final.md`의 다수 항목을 코드상 반영했지만, `require.md` 제출 기준으로는 아직 PASS가 아니다.

가장 큰 차단 사유는 다음 네 가지다.

1. 현재 산출물이 full requirement evidence가 아니라 small-mode evidence다.
2. cohort/journey 산출물이 `require.md`의 M1/M3/M6/M12, churn-last-30 top-5, pre-churn event, journey funnel 요구를 만족하지 못한다.
3. recommendation 산출물이 sleeping dog 또는 negative uplift 고객에게 no-action을 보장하지 못한다.
4. pipeline resume/cache, dashboard fallback, stale `data/artifacts` 때문에 존재하는 파일만으로 제출 조건 충족을 증명하기 어렵다.

## 요구사항별 상태

### 1. 고객 행동 시뮬레이터

부분 충족.

확인된 긍정 증거:
- `config/simulator_config.yaml`은 6개 persona, 8개 이상 event type, configurable churn definition, churn target 15~25%, treatment/control 설정을 포함한다.
- 현재 small artifact는 5,000 customers, 180 days, treatment/control 2,500/2,500, churn rate 약 19.12%로 small-mode 자체는 맞다.

남은 이슈:
- 현재 workspace evidence는 full mode가 아니다. `data/raw/generation_summary.json` 기준 5,000명/2,500 treatment/2,500 control이다.
- full 제출 기준인 20,000 customers, 12개월 이상, treatment/control 각각 10,000명 이상을 현재 산출물로 증명하지 못한다.
- `data/raw/pipeline_state.json`이 small run 완료 상태를 들고 있어 full run이 stale state를 재사용할 수 있다.

### 2. 코호트 및 고객 여정 분석

미충족.

남은 이슈:
- `results/cohort_retention_matrix.csv`가 period `0.0`만 가진다.
- `results/cohort_milestones.csv`의 M1/M3/M6/M12 값이 비어 있다.
- `results/cohort_analysis.json`이 `available_milestones: []`를 기록한다.
- `results/churn_last30_sequences.json`이 빈 배열이다.
- 최신 `cohort_analysis.json`은 `pre_churn_events_error: "'event_type'"`, `journey_funnel_error: "'event_type'"`를 기록한다.
- 오래된 `pre_churn_events.csv`, `journey_funnel.csv`가 남아 있어 artifact existence check가 실제 실패를 가릴 수 있다.

필요 조치:
- cohort 입력 이벤트 스키마를 `event_type` 기준으로 정규화하거나 analyzer가 현재 raw event 컬럼명을 받아들이도록 수정한다.
- M1/M3/M6/M12를 항상 산출하고, 관측 기간 부족 시 명시적 policy를 둔다.
- churn 고객 마지막 30일 sequence top-5를 비어 있지 않은 JSON/CSV로 저장한다.
- pre-churn event 빈도와 journey funnel 산출 실패 시 stale 파일을 제거하거나 checklist에서 실패 처리한다.

### 3. 피처 엔지니어링

대체로 충족.

확인된 긍정 증거:
- `docs/feature_dictionary.md`가 33개 feature를 문서화한다.
- `results/features.csv`는 5,000 rows, 38 columns이며 RFM/change/session/sequence/time/journey 계열 feature를 포함한다.
- `data/feature_store/features.csv`와 `.parquet`가 존재하고 `requirements.txt`에 `pyarrow`가 포함됐다.

남은 이슈:
- outlier capping/처리 evidence가 약하다. 결측/inf 처리는 보이지만 제출 설명용으로 이상치 처리 로직을 더 명확히 드러낼 필요가 있다.
- `_compute_features()`가 기본 `results/features.csv`를 우선 재사용하는 경로는 stale small features 재사용 리스크가 있다.

### 4. ML 기반 이탈 예측

Scoped PASS.

확인된 긍정 증거:
- LightGBM/XGBoost 비교, 5-fold CV, tuning evidence가 있다.
- `results/model_metrics.json`은 ML/DL/ensemble 비교를 포함한다.
- `results/shap_summary.png`, `results/feature_importance.csv`가 존재한다.
- Ramanujan agent는 이 관점에서 blocker 없음으로 판정했다.

잔여 리스크:
- 현재 수치 evidence는 small-mode artifact다.
- AUC 1.0 계열 수치는 synthetic leakage나 너무 쉬운 target 가능성을 후속 검토할 필요가 있다.

### 5. DL 기반 이탈 예측

Scoped PASS.

확인된 긍정 증거:
- `results/dl_training_log.json`이 `sequence_source: event_sequence`, `architecture: lstm`, `best_epoch`를 기록한다.
- `models/dl_churn_model.pt`가 존재한다.
- ML/DL/ensemble 비교가 `model_metrics.json`과 performance history에 기록된다.

잔여 리스크:
- 현재 evidence는 small-mode run이다.

### 6. Uplift Modeling

Scoped PASS.

확인된 긍정 증거:
- `results/uplift_learner_comparison.csv`가 `s_learner`, `t_learner`를 비교한다.
- `results/uplift_results.csv`가 5,000 rows, `uplift_score`, `treatment_effect`, `baseline_churn_probability`, `segment`를 포함한다.
- 4분면 uplift segment가 모두 존재한다.
- `results/qini_curve.png`가 존재한다.

잔여 리스크:
- 비교상 `s_learner`가 더 높지만 기본 output selected learner는 `t_learner`로 보인다. selection policy를 명확히 해야 한다.
- 현재 treatment/control evidence는 small-mode 2,500/2,500이다.

### 7. CLV 예측

Scoped PASS.

확인된 긍정 증거:
- `results/clv_predictions.csv`가 5,000 rows를 가진다.
- top-20% flag가 1,000 customers로 산출된다.
- `results/clv_actual_vs_predicted.csv`, `results/clv_validation.json`, `results/clv_top_customers.csv`가 존재한다.

잔여 리스크:
- validation target이 실제 미래 매출이 아니라 `monetary_12m_proxy` 성격이다. 문서와 코드에서 이 한계를 명시해야 한다.

### 8. 고객 세그먼테이션 및 우선순위

부분 충족.

확인된 긍정 증거:
- `results/segments_6plus.csv`가 5,000 rows, 6 operating segments를 가진다.
- `priority_score == uplift_score * clv` 관계가 확인됐다.

남은 이슈:
- high-risk/high-value/positive-uplift 조합이 현재 artifact에서 0건으로 확인됐다.
- 고가치 Persuadable 또는 고가치 Lost Cause segment evidence가 약하다.

### 9. 리텐션 전략 및 예산 최적화

부분 충족.

확인된 긍정 증거:
- `results/budget_whatif.csv`는 50/100/200 scenarios를 가진다.
- `results/budget_optimization_summary.json`은 ROI를 기록한다.
- sleeping dog 및 negative uplift 예산은 0으로 처리된다.

남은 이슈:
- recommendation layer에서는 동일한 no-action policy가 지켜지지 않는다.
- `results/recommendations.csv`가 negative uplift/sleeping dog 고객에게 active action을 부여한다.
- recommendation schema가 `customer_id, action_type, score, estimated_cost, reason` 중심이라 segment/uplift/CLV/churn/expected ROI/no-action evidence가 부족하다.

필요 조치:
- `run_recommend`가 `segments_6plus.csv`와 `uplift_results.csv`를 결합해 추천해야 한다.
- `uplift_score <= 0` 또는 uplift segment `sleeping_dog`는 `no_action` 또는 exclusion으로 처리해야 한다.
- recommendation artifact에 `segment`, `uplift_segment`, `uplift_score`, `clv`, `churn_probability`, `priority_score`, `expected_roi`, `expected_revenue_saved`를 포함해야 한다.

### 10. A/B 테스트

부분 충족.

확인된 긍정 증거:
- `results/ab_test_results.json`은 p-value, 95% CI, lift, required sample size를 포함한다.
- `results/ab_test_detailed.json`은 Cohen's h와 power를 포함한다.

남은 이슈:
- 현재 artifact는 small-mode 2,500/2,500 기준이라 full requirement의 10,000/10,000 evidence가 아니다.
- detailed observed power가 0.69, 0.73으로 문서상 0.80 target보다 낮다.

### 11. 통합 대시보드

부분 충족.

확인된 긍정 증거:
- dashboard import와 page registration은 통과했다.
- loader 기본 순서는 `results/` 우선으로 개선됐다.
- 주요 artifact schema adapter가 상당 부분 존재한다.

남은 이슈:
- cohort dashboard가 incomplete current output을 숨기고 computed/sample fallback을 표시할 수 있다.
- churn overview는 전체 고객이 아니라 test/ensemble subset 834 rows만 기준으로 저장된다.
- `data/artifacts` mirror가 stale/mismatched 상태다. 예: `data/artifacts/budget_optimization.csv`는 1 synthetic row, `data/artifacts/recommendations.csv`는 10 rows지만 `results/`는 5,000 rows다.

필요 조치:
- dashboard fallback은 제출 필수 artifact에 대해서는 silent sample fallback이 아니라 warning/fail visible state로 바꾼다.
- churn prediction artifact를 전체 customers 기준으로 저장하거나 dashboard copy에서 subset임을 명확히 분리한다.
- artifact mirror를 pipeline 종료 시 강제 동기화하거나 dashboard에서 stale artifact를 거부한다.

### 12. 모델 모니터링

부분 충족.

확인된 긍정 증거:
- `results/monitoring_report.json`은 PSI/KS feature alerts 33개, `overall_alert_level`, performance latest rows를 포함한다.

남은 이슈:
- performance time-series loader가 `results/model_performance_history.csv` 파일명을 찾지 않아 sample/MLflow fallback으로 갈 수 있다.

필요 조치:
- dashboard data loader의 metric history search path에 `model_performance_history.csv`를 포함한다.
- AUC/Precision/Recall time-series가 current artifact에서 직접 표시되는지 확인한다.

### 13. 문서화 및 코드 품질

부분 충족.

확인된 긍정 증거:
- `docs/requirement_traceability.md`가 `issue_final.md` 항목별 반영 위치를 기록한다.
- README/docs 다수가 현재 pipeline 중심으로 업데이트됐다.
- `docker compose config --quiet`, CLI import/help, `git diff --check`, Python AST parse는 통과했다.

남은 이슈:
- 현재 traceability evidence가 small E2E와 full tests 중심이다. full-mode artifact evidence가 없다.
- tracked generated files와 pycache가 dirty 상태다.
- `results/`, `data/`, 대부분 model outputs는 ignored라 제출은 reproducible full run으로 증명해야 한다.

## 구현 우선순위

1. Cohort/journey artifact를 먼저 고친다.
   - M1/M3/M6/M12 milestone, churn-last-30 top-5, pre-churn event, journey funnel을 실제 current event schema로 생성한다.
   - 실패한 cohort analysis가 stale CSV/JSON으로 가려지지 않도록 checklist를 강화한다.

2. Pipeline cache/resume와 full-run evidence 리스크를 고친다.
   - `--small` 여부, customer count, input/output path가 pipeline state에 반영되도록 한다.
   - full mode에서 stale small artifact를 reuse하지 않게 한다.
   - `_compute_features()`가 `--data`/`--output` context를 무시하지 않게 한다.

3. Recommendation no-action policy를 고친다.
   - negative uplift/sleeping dog 고객은 active recommendation을 받지 않도록 한다.
   - recommendation CSV를 final decisioning schema로 확장한다.

4. Dashboard fallback과 monitoring history wiring을 고친다.
   - 필수 artifact가 invalid하면 sample fallback 대신 visible warning/fail state를 반환한다.
   - `model_performance_history.csv`를 AUC/Precision/Recall history source로 연결한다.

5. Repository hygiene를 정리한다.
   - tracked pycache와 generated model dirty state를 제거한다.
   - `.audit/` 같은 임시 산출물은 commit 대상에서 제외한다.
   - commit에는 요구사항 반영 코드, 테스트, docs, issue docs만 포함한다.

6. 마지막에 무컨텍스트 검증 agent를 다시 통과시킨다.
   - 이번 6-agent 검증은 PASS 조건을 만족하지 못했다.
   - 모든 blocker 수정 후 새 contextless verifier가 `require.md`와 `issue_final_v2.md` 기준으로 PASS해야 한다.

## 이번 검증에서 테스트 실행 여부

사용자가 "테스트가 다 돈다는 것은 확인해서 테스트는 안 돌아도 된다고 생각"한다고 지시했으므로 이번 6-agent 검증에서는 full test suite를 새로 실행하지 않았다.

기존 기록상 전체 테스트 evidence는 다음과 같다.

```bash
OMP_NUM_THREADS=1 LIGHTGBM_NUM_THREADS=1 /tmp/capstone-codex-py312/bin/python -m pytest -q
# 2503 passed, 85 warnings
```

단, 위 테스트 통과는 현재 6-agent 검증에서 발견한 artifact completeness, small/full evidence, recommendation no-action, dashboard fallback 문제를 자동으로 보증하지 않는다.

## 후속 구현 반영 상태

`issue_final_v2.md` 작성 및 커밋 이후, 위 FAIL 항목 중 코드와 small-mode artifact에서 즉시 검증 가능한 항목을 보강했다.

반영 위치:
- Cohort/journey: `src/analysis/cohort_analysis.py`, `src/main.py::run_cohort`
- Recommendation no-action: `src/models/recommendations.py`, `src/main.py::run_recommend`
- Pipeline cache/resume 및 checklist validation: `src/main.py::_compute_features`, `src/main.py::run_all`, `src/main.py::_write_artifact_checklist`
- Dashboard performance history/fallback: `src/dashboard/data_loader.py`
- 반영 기록: `docs/requirement_traceability.md`

현재 focused evidence:
- `results/required_artifacts_checklist.json`: `25 / 25`, `missing: []`
- `results/cohort_milestones.csv`: M1/M3/M6/M12 populated
- `results/churn_last30_sequences.json`: top-5 sequence patterns populated
- `results/recommendations.csv`: no-action 대상 고객의 active recommendation 위반 `0`
- Targeted tests: recommendations `33 passed`, dashboard/streamlit/pipeline `319 passed`, main/CLI/pipeline `162 passed`, budget/drift/monitoring `172 passed`

남는 제출 리스크:
- 현재 생성 산출물은 여전히 small-mode evidence다. full-mode 20,000명/12개월/10,000+ treatment-control 증명은 최종 제출 직전 별도 clean full run으로 확인해야 한다.

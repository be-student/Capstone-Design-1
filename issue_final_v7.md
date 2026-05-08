# issue_final_v7.md

## 목적

이번 문서는 `issue_final_v6.md` 이후 사용자의 요청에 따라 `require.md`의 모든 요구사항이 구현되어 있는지 다시 확인한 v7 검증 사이클의 결과를 기록한다.

사용자 지시:
- `require.md`의 요구사항 전체 확인
- 명세를 12개 조각으로 분해
- 6개 agent에게 2번에 나누어 검증 위임
- 메인 컨텍스트는 오케스트레이션만 수행

운영 원칙:
- 메인 컨텍스트는 `require.md`, `issue_final_v6.md`, git 상태 확인, agent 분해/취합, 결과 문서 작성만 수행했다.
- 코드 구현과 수정을 하지 않았다.
- 모든 verifier는 fresh-context read-only 검증으로 실행했다.

## 12개 검증 조각

| 조각 | 범위 | 담당 | 판정 |
| --- | --- | --- | --- |
| 1 | 최종 실행/산출물 계약 | verifier | PASS |
| 2 | 고객 행동 시뮬레이터 | verifier | FAIL |
| 3 | 코호트 및 고객 여정 분석 | verifier | FAIL |
| 4 | 피처 엔지니어링 | verifier | PASS |
| 5 | ML 기반 이탈 예측 모델 | verifier | PASS |
| 6 | DL 기반 이탈 예측 모델 | verifier | PASS |
| 7 | Uplift Modeling | verifier | PASS |
| 8 | CLV 예측 | verifier | PASS |
| 9 | 고객 세그먼테이션 및 우선순위 | verifier | PASS |
| 10 | 리텐션 전략/예산 최적화 및 A/B 테스트 | verifier | FAIL |
| 11 | 통합 대시보드 및 모델 모니터링 | verifier | FAIL |
| 12 | 제약 사항/문서화/코드 품질/보너스 통합 | verifier | FAIL |

종합: 7 PASS / 5 FAIL.

## PASS로 확인된 주요 항목

1. 최종 실행/산출물 계약
   - `docker compose config --quiet` 통과.
   - `src/main.py --mode train/uplift/optimize --budget` CLI가 존재한다.
   - `results/required_artifacts_checklist.json` 기준 `29/29`, `missing: []`, `full_submission_ready: true`.

2. 피처 엔지니어링
   - RFM, 5개 이상 행동 변화율, 구매 주기 이상, 세션 품질, 시퀀스, 시간대별 행동, 여정 단계/체류 기간 피처가 구현되어 있다.
   - `docs/feature_dictionary.md`에는 30개 이상 피처와 비즈니스 의미가 있다.
   - `data/feature_store/features.csv`와 parquet feature store가 존재한다.

3. ML/DL 모델
   - LightGBM/XGBoost 2개 트리 모델 비교, class imbalance 처리, 5-fold CV, AUC >= 0.78, SHAP summary/local explanation, threshold analysis, hyperparameter grid가 확인되었다.
   - PyTorch 2.x 기반 LSTM/Transformer 구조, sequence padding/projection, early stopping, ML/DL/ensemble 비교, CPU 테스트가 확인되었다.

4. Uplift/CLV/Segmentation
   - Treatment/Control 10,000/10,000 데이터를 사용한다.
   - T-Learner/S-Learner 비교, 고객별 uplift score, 4분면 세그먼트, Qini curve, Persuadables 분석이 확인되었다.
   - CLV 12개월 예측, 상위 20% high-value flag, actual vs predicted 검증, top customer/distribution report가 확인되었다.
   - 6개 세그먼트, segment summary, `priority_score = uplift_score * clv`, 시각화가 확인되었다.

## 필수 blocker

최종 판정이 PASS가 아닌 이유는 아래 blocker 때문이다.

1. 시뮬레이터 산출물 스키마 불일치
   - 코드와 테스트는 `session_duration`, `marketing_channel`, `marketing_response`를 다루지만 현재 full-mode 제출 산출물 `data/raw/events.csv` / `events.parquet`에는 해당 컬럼이 없다.
   - 따라서 현재 생성 데이터 산출물만으로 시간 경과 세션 시간 변화와 페르소나별 마케팅 반응을 검증할 수 없다.
   - verifier evidence: raw events header는 `customer_id,event_type,event_date,timestamp,amount`뿐이며, tests는 `54 passed`.

2. 코호트 이탈률 차이 시각화 누락
   - `results/cohort_churn_rates.csv`는 존재하지만, 코호트별 이탈률 차이를 직접 시각화한 results 파일이 확인되지 않았다.
   - `require.md`는 코호트별 이탈률 차이를 분석하고 시각화해야 한다고 요구한다.

3. 고객 여정 퍼널의 이탈 시점 분석 evidence 누락
   - `results/journey_funnel.csv`는 `stage,count,conversion_rate,drop_off_rate`만 포함한다.
   - 가입 -> 첫구매 -> 재구매 -> 충성 -> 이탈 퍼널의 전환율은 있으나, 이탈 시점 또는 단계별 소요 시간/tenure 관련 산출물이 없다.

4. A/B 테스트 교란 통제 및 밸런스 체크 근거 누락
   - A/B Power Analysis, p-value, 95% CI, z-test는 통과했다.
   - 하지만 `docs/ab_test_report.md`와 산출물에서 교란 통제 전략 선택 이유, covariate/persona balance-check 방법, persisted balance-check evidence가 확인되지 않았다.

5. 모델 모니터링의 성능 저하 alert 누락
   - PSI/KS drift detection, drift alert, `results/monitoring_report.json`의 threshold 기록은 확인되었다.
   - 하지만 AUC/Precision/Recall의 시간별 성능 저하를 threshold와 비교해 alert로 생성하는 구현 또는 report 필드가 확인되지 않았다.

6. 필수 제약/문서화 gap
   - `require.md` 제약 사항의 "Git Flow 또는 GitHub Flow 브랜치 전략" 사용/문서화 evidence가 없다.
   - `rg "Git Flow|GitHub Flow|branch|브랜치|pull request|feature branch" README.md docs src scripts tests`에서 관련 문서가 확인되지 않았다.

## 권장/선택 항목 gap

아래는 `require.md`의 구현 가이드 또는 보너스 영역이므로 필수 blocker와 분리한다. 다만 제출 품질 관점에서는 후속 조치가 필요하다.

1. `src/data/make_dataset.py`가 없다.
   - 구현 가이드 G4는 `src/data/make_dataset.py` 실행으로 데이터 생성부터 processed dataset 저장까지 재현 가능하게 만들 것을 권장한다.

2. 모델 파일명이 versioned naming을 따르지 않는다.
   - 현재 `ml_churn_model.pkl.joblib`, `dl_churn_model.pt` 등 고정 파일명이 사용된다.

3. `ruff check src --no-cache --statistics`에서 86개 style error가 보고되었다.
   - unused import/variable, placeholder 없는 f-string, E402 등이 포함된다.

4. MLflow 보너스는 서비스/테스트 구조는 있으나 실제 `mlruns` artifact가 없고 `run_all`에서 `mlflow_logging`이 `_noop`으로 기록된다.
   - 보너스 4는 선택 과제이므로 필수 FAIL 원인은 아니다.

## Agent 검증 명령 요약

각 verifier가 실행한 대표 명령:

- `docker compose config --quiet`
- `pytest tests/test_cli_entrypoint.py tests/test_main_cli.py tests/test_pipeline_runner.py tests/test_docker_setup.py` -> `320 passed`
- `pytest tests/test_data_generator.py` -> `54 passed`
- `pytest tests/test_cohort_analysis.py tests/test_cohort_computations.py tests/test_pipeline_runner.py` -> `164 passed`
- `pytest tests/test_feature_engineering.py` -> `36 passed`
- `pytest tests/test_churn_model.py tests/test_shap_explainer.py` -> `68 passed, 1 skipped`
- `pytest tests/test_sequence_dataset.py tests/test_dl_trainer.py` -> `45 passed`
- `pytest tests/test_uplift.py tests/test_uplift_model.py` -> `57 passed`
- `pytest tests/test_clv.py tests/test_clv_model.py` -> `64 passed`
- `pytest tests/test_segmentation.py` -> `34 passed`
- `pytest tests/test_budget_optimization.py tests/test_budget_lp_solver.py tests/test_budget_optimizer.py tests/test_ab_testing.py tests/test_ab_statistical_methods.py` -> `208 passed`
- `pytest tests/test_streamlit_dashboard.py tests/test_dashboard.py tests/test_model_monitoring_view.py tests/test_monitoring_service.py tests/test_ks_drift.py tests/test_docker_setup.py` -> `564 passed`

## 최종 판정

최종 판정: FAIL.

`require.md`의 대부분의 기능은 구현되어 있고 많은 테스트가 통과하지만, fresh-context verifier 12개 중 5개가 FAIL을 반환했다. 특히 simulator 제출 산출물 스키마, 코호트/여정 산출물, A/B 교란 통제 evidence, 성능 저하 alert, GitHub Flow/Git Flow 문서화 gap은 필수 요구사항의 PASS를 막는다.

다음 사이클은 위 필수 blocker를 닫는 구현 작업으로 시작해야 한다. 메인 컨텍스트는 계속 오케스트레이션만 수행하고, 각 blocker를 non-overlapping write scope로 나누어 worker에게 위임해야 한다.

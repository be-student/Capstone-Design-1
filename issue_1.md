# Issue 1 - 남은 요구사항 정리

## 검토 범위

- 기준 문서: `require.md`
- 방식: 명세를 12개 범위로 나누고, Codex native agent 6개씩 2회 read-only 감사 후, 6개 read-only 리뷰어가 근거 품질을 재검토함.
- 제한: 코드 수정, 테스트 실행, `docker compose up` 실행은 하지 않음.

## 12등분 감사 분배

1. 최종 산출물, Docker Compose, CLI, 실행 제약
2. 고객 행동 시뮬레이터와 데이터 제약
3. 코호트 및 고객 여정 분석
4. 피처 엔지니어링과 feature store
5. ML churn 모델과 SHAP
6. DL sequence 모델과 ensemble
7. Uplift modeling과 4분면/6세그먼트 연계
8. CLV 예측
9. 리텐션 전략 및 예산 최적화
10. A/B 테스트 설계 및 분석
11. 통합 대시보드와 모델 모니터링
12. 문서화, 코드 품질, 보너스 과제

## 현재 산출물 상태

- 현재 존재: `results/budget_optimization.csv`, `results/cohort_analysis.json`, `results/cohort_retention_curves.png`, `results/cohort_retention_heatmap.png`, `results/cohort_retention_matrix.csv`, `models/ml_churn_model.pkl.joblib`
- 필수 산출물 중 누락 확인: `results/model_metrics.json`, `results/shap_summary.png`, `results/clv_predictions.csv`, `results/segments_6plus.csv`, `results/monitoring_report.json`, `models/dl_churn_model.pt`
- 제출 전 `docker compose up --build`, 핵심 CLI 모드, pytest/ruff, 필수 산출물 생성 여부를 다시 검증해야 함.

## P0 - 제출/실행 차단 이슈

- [ ] `docker compose up` 기본 실행이 과제 full 조건과 맞는지 정리해야 함. 현재 `docker-compose.yml`은 `SMALL=${SMALL:-true}`라 기본이 small mode임. 근거: `docker-compose.yml:110-111`, `require.md:71`.
- [ ] `run_all` 파이프라인에 필수 산출 단계가 빠져 있음. 주석상 순서에는 `segment`, `cohort`가 있지만 실제 `step_handlers`에는 둘 다 없음. 근거: `src/main.py:762-804`.
- [ ] Parquet 엔진 의존성을 추가해야 함. 코드가 `to_parquet`/`read_parquet`를 쓰지만 `requirements.txt`에 `pyarrow` 또는 `fastparquet`가 없음. 근거: `src/data/orchestrator.py:229-233`, `src/features/feature_engineering.py:798-823`.
- [ ] 대시보드 단독 실행/스킵 흐름을 정리해야 함. `SKIP_PIPELINE` 환경변수는 있으나 dashboard가 여전히 `pipeline: service_completed_successfully`에 묶임. 근거: `docker-compose.yml:160-177`.
- [ ] 파이프라인 산출물 경로와 대시보드 로더 경로를 통일해야 함. 파이프라인은 `results/`에 저장하고, `DashboardDataLoader`는 기본 `data/artifacts`를 읽음. 근거: `src/dashboard/data_loader.py:20`, `src/main.py:295`, `src/main.py:731`.

## 시뮬레이터/데이터

- [ ] `treatment.min_group_size: 10000`을 실제로 강제해야 함. 설정에는 있지만 생성기는 `treatment_ratio` 기반 Bernoulli 배정만 수행함. 근거: `config/simulator_config.yaml:42-44`, `src/data/generator.py:63-64`, `src/data/generator.py:126-127`.
- [ ] 목표 churn rate 15-25%를 생성 후 검증하거나 calibration/fail 처리해야 함. 설정은 읽지만 생성 로직에서 보정/실패 조건으로 사용하지 않음. 근거: `config/simulator_config.yaml:35`, `src/data/generator.py:66-67`, `src/data/orchestrator.py:164-195`.
- [ ] 런타임 config schema/range validator를 추가해야 함. `load_config`는 YAML 로드 중심이고 generator는 키 접근에 의존함. 검증 대상: persona required keys, 확률 범위, persona 비율 합, 이벤트 타입 개수, 기간/고객 수.
- [ ] `purchase_frequency_monthly`, `avg_session_minutes` 같은 persona 설정을 실제 이벤트 생성에 반영하거나 문서/설정에서 제거해야 함.
- [ ] `Preprocessor` 검증을 생성/파이프라인 경로에 자동 연결해야 함. 검증 메서드는 있으나 `run_simulate`, `SimulatorOrchestrator`, `run_all`에서 자동 호출되지 않음. 근거: `src/data/preprocessing.py:366-432`.

## 코호트/피처

- [ ] 코호트 결과에서 M1/M3/M6/M12를 명시적으로 추출하고 라벨링해야 함. 현재는 전체 period matrix/plot 저장 중심임. 근거: `require.md:75-81`, `src/main.py:602`.
- [ ] 이탈 고객의 마지막 30일 패턴과 이탈 직전 이벤트 분석을 CLI/대시보드/테스트/결과 저장에 연결해야 함. 함수는 있으나 `run_cohort` 산출 경로에 연결되지 않음.
- [ ] 고객 여정 퍼널 분석을 산출물로 저장하고 테스트해야 함. `compute_journey_funnel`은 있으나 `run_cohort`/대시보드 연결이 부족하고, `Churned` 단계의 비단조 퍼널 정의를 점검해야 함.
- [ ] `run_features`가 전용 feature store를 사용하도록 연결해야 함. `FeatureEngineer.save_to_feature_store()`는 있으나 CLI는 `results/features.csv`만 저장함. 근거: `src/features/feature_engineering.py:784`, `src/main.py:666`.
- [ ] 문서가 언급하는 `src/features/store.py` 또는 Redis hash 기반 feature store를 구현하거나 문서를 정정해야 함. 현재 `src/features`에는 전용 store 모듈이 없음.
- [ ] `feature_store` YAML 설정을 추가하고 저장 경로/형식을 config 기반으로 연결해야 함.
- [ ] sequence embedding 요구를 충족하려면 실제 이벤트 시퀀스 벡터화/embedding을 추가해야 함. 현재는 `sequence_diversity`, `sequence_length`, cluster, trend 같은 집계 피처 중심임.

## ML/DL

- [ ] `run_train`의 SHAP 통합을 수정해야 함. `ShapExplainer` 생성자는 background data를 요구하고, main은 존재하지 않는 `summary_plot`, `top_features` 호출을 사용함. 근거: `src/main.py:283-286`, `src/models/shap_explainer.py:42`, `src/models/shap_explainer.py:141`.
- [ ] threshold analysis를 학습 결과에 연결해야 함. `analyze_threshold()`는 있으나 `model_metrics.json`에 optimal threshold/precision/recall trade-off가 저장되지 않음. 근거: `src/models/churn_model.py:563`, `src/main.py:256`.
- [ ] `model_metrics.json` 스키마를 대시보드 기대와 맞춰야 함. 대시보드는 `ml_model/dl_model/ensemble` 및 `auc/f1_score` 형태를 기대하고, `run_train`은 `ml_metrics/dl_metrics/ensemble_metrics` 및 `auc_roc/f1` 형태를 저장함.
- [ ] DL 학습 입력을 실제 고객 행동/event sequence로 연결해야 함. 현재 CLI train 경로는 tabular feature row를 반복한 pseudo-sequence를 사용하고, `sequence_utils.create_sequences()`는 main 학습 경로에 연결되지 않음.
- [ ] CLI `--mode train`에서 `DLTrainer`의 EarlyStopping과 LSTM/Transformer architecture selection을 실제로 사용해야 함. 현재는 `DLChurnModel.fit()` 직접 호출. 근거: `src/main.py:267`, `src/models/dl_trainer.py:328-399`.
- [ ] `dl_model_training`, `ensemble_creation` 파이프라인 step을 no-op이 아니라 독립 산출물/상태를 남기도록 정리해야 함. 근거: `src/main.py:793-795`.

## Uplift/Segmentation

- [ ] `run_uplift` 입력 검증을 강화해야 함. `treatment_group` 누락 시 전원 treatment, `churn_label` 누락 시 전원 0으로 fallback함. 근거: `src/main.py:309-311`.
- [ ] 최소 2가지 uplift learner를 같은 데이터에서 구현/비교해야 함. 현재 CLI는 단일 `--learner`만 선택해 한 모델만 학습함. 근거: `require.md:114`, `src/main.py:319-321`.
- [ ] 4분면 분류에 기본 이탈 확률 축을 포함해야 함. 현재 `UpliftModel.segment_customers()`는 `uplift_scores`만 받아 median/0 기준으로 분류함. 근거: `require.md:116-117`, `src/models/uplift_model.py:197-218`.
- [ ] Qini/AUUC의 부호와 uplift 정의를 정리해야 함. 문서는 `control churn - treatment churn` 방향인데 계산은 churn outcome 기준 부호가 뒤집힐 수 있고 `abs()`로 숨김. 근거: `docs/uplift_analysis.md:20`, `src/models/uplift_model.py:295-310`.
- [ ] Persuadables 특성 분석과 타겟팅 기준을 실제 산출물 기반으로 생성해야 함. 현재 uplift output은 `customer_id/uplift_score/segment` 중심이고 문서는 정적 설명임. 근거: `require.md:122`, `src/main.py:329`.
- [ ] `segments_6plus.csv`를 churn probability, uplift, CLV 결합 기준으로 재구현해야 함. 현재 `run_segment`는 RFM/KMeans 중심이며 uplift/CLV 결과를 병합하지 않음. 근거: `require.md:130`, `src/features/segmentation.py`, `src/main.py:643`.
- [ ] 4분면 x CLV x 이탈확률 기반 6세그먼트 테스트를 추가해야 함. 현재 테스트는 uplift 양수/segment 개수 등 부분 검증 중심임.

## CLV

- [ ] `run_clv`에 holdout 기반 실제 vs 예측 검증을 추가해야 함. 현재는 같은 `X`로 학습하고 같은 `X`로 예측함. 근거: `require.md:127`, `src/main.py:358`, `src/main.py:368`.
- [ ] 향후 12개월 CLV 산출을 관측기간/예측기간으로 분리해야 함. 현재는 전체 구매 이벤트 합계 기반 `monetary * 12` proxy임. 근거: `require.md:125`, `src/features/feature_engineering.py:147`, `src/main.py:358`.
- [ ] `clv_predictions.csv` 외 Top-N, 분포, actual-vs-predicted 리포트/플롯을 생성해야 함. 근거: `require.md:128`, `src/main.py:377`, `docs/models.md:699`, `docs/models.md:745`.
- [ ] CLV 산출물 스키마를 통일해야 함. CLI는 `predicted_clv/high_value`, 대시보드는 `clv_predicted`와 `clv_data.csv` 흐름을 기대함. 근거: `src/main.py:372`, `src/dashboard/data_loader.py:138-144`, `src/dashboard/app.py:1938`.
- [ ] CLV 문서를 구현과 맞춰야 함. 요구사항은 ML 기반도 허용하지만, 일부 README/docs는 lifetimes/BG-NBD/Gamma-Gamma를 실제 구현처럼 설명하고 코드는 `GradientBoostingRegressor` 기반임. 근거: `src/models/clv_model.py:47`, `README.md:168`, `docs/models.md:640`.

## Budget/A-B Test

- [ ] `run_optimize` 결과에 expected revenue/ROI를 저장해야 함. 현재 CSV와 반환값은 allocated budget 중심임. 근거: `require.md:143`, `src/main.py:447-448`.
- [ ] CLI `run_optimize`에 50%/100%/200% what-if 저장/반환을 연결해야 함. 현재 단일 예산 최적화만 실행함. 근거: `require.md:142`, `src/main.py:441-442`.
- [ ] `src/optimization` 기본 sweep을 50%/100%/200%로 맞춰야 함. 현재 기본값은 50%/100%/150%임. 근거: `src/optimization/budget_optimizer.py:1210-1216`.
- [ ] 고객별 처우 상한과 ROI 계산을 안정화해야 함. 큰 예산에서 처우 상한이 완화될 수 있고, `allocated / cost`가 1을 초과하면 retained value/ROI가 과대 산출될 수 있음. 근거: `src/models/budget_optimizer.py:178-185`, `src/models/budget_optimizer.py:682-691`.
- [ ] 비용 구조를 문서, config, CLI에서 일관되게 연결해야 함. 문서 비용, config 채널 비용, `run_optimize` 하드코딩 비용이 다름. 근거: `docs/retention_strategy.md:51`, `config/simulator_config.yaml:161`, `src/main.py:421`.
- [ ] `run_optimize` 입력을 실제 예측 이탈확률로 연결해야 함. 현재 upstream 데이터가 있어도 `customers["churn_label"]`을 `churn_prob`로 사용함. 근거: `src/main.py:409-420`.
- [ ] `run_ab_test`가 simulator 결과를 필수 입력으로 사용하도록 해야 함. 현재 컬럼 누락 시 합성 5,000/5,000 fallback을 생성함. 근거: `require.md:145`, `src/main.py:472`.
- [ ] 실제 실행 결과 기반 A/B 해석 리포트를 생성하거나 `docs/ab_test_report.md` 갱신 흐름을 추가해야 함. 현재 CLI는 JSON만 저장하고 문서는 정적 예시 중심임. 근거: `require.md:150`, `require.md:172`, `src/main.py:500`, `docs/ab_test_report.md:232`.
- [ ] A/B 결과 JSON 스키마를 대시보드 A/B view 기대와 맞춰야 함. 현재 산출 JSON과 dashboard detailed schema가 연결되지 않음. 근거: `src/main.py:483-500`, `src/dashboard/data_loader.py:72`.
- [ ] churn 감소 실험의 효과 방향을 정리해야 함. Power 계산은 `baseline + mde` 증가 방향이고 문서 예시는 이탈률 감소임. 근거: `src/models/ab_testing.py:69`, `docs/ab_test_report.md:227`.
- [ ] churn metric에서 유의성 해석/추천 문구가 개선 방향을 반대로 볼 수 있는 부분을 수정해야 함. 근거: `src/analysis/ab_testing.py:1109`.

## Dashboard/Monitoring

- [ ] 고객별 이탈 위험 분포용 `churn_predictions.csv` 또는 adapter를 생성해야 함. 대시보드는 `churn_probability`를 기대하지만 `run_train`은 고객별 예측 CSV를 만들지 않음. 근거: `src/dashboard/data_loader.py:51`, `src/dashboard/app.py:109-118`, `src/main.py:261-295`.
- [ ] 예산/CLV 대시보드 입력 스키마를 실제 output과 맞춰야 함. 대시보드는 `budget_results.csv/allocated_budget_krw`, `clv_data.csv/clv_predicted`를 기대하지만 파이프라인은 `budget_optimization.csv/allocated_budget`, `clv_predictions.csv/predicted_clv`를 저장함.
- [ ] `app.py`의 `render_model_monitoring` shadowing을 정리해야 함. `monitoring_view.render_model_monitoring`을 import한 뒤 같은 이름의 로컬 함수가 다시 정의되어 성능 추적 섹션이 사용되지 않을 수 있음. 근거: `src/dashboard/app.py:31`, `src/dashboard/monitoring_view.py:64`, `src/dashboard/app.py:4017`, `src/dashboard/app.py:4745`.
- [ ] `run_monitor`가 `feature_alerts`를 읽도록 수정해야 함. `DriftReport`/`KSDriftReport`는 `feature_alerts`를 갖는데 `run_monitor`는 `.alerts` 속성을 검사해 alert 배열이 비게 됨. 근거: `src/main.py:713-716`, `src/monitoring/drift_detection.py:172`, `src/monitoring/ks_drift.py:153`.
- [ ] AUC/Precision/Recall의 시간별 성능 추적 산출물을 추가해야 함. 현재 drift 중심이고 대시보드는 현재 metric/MLflow/sample run 표시 위주임.
- [ ] drift/성능 저하 알림을 CLI report와 대시보드에서 확인 가능한 형태로 남겨야 함. 현재는 callback/MLflow 중심이고 CLI report에는 alert 상세/성능 저하 알림이 충분히 남지 않음.
- [ ] dashboard/monitoring 테스트를 실제 `results/` 파일명과 스키마 회귀를 검증하도록 보강해야 함. 현재는 sample/fallback fixture 중심임.

## Docs/Quality/Bonus

- [ ] README 상세 구조와 일부 문서 예제를 실제 코드 구조와 맞춰야 함. 예: `realtime_scorer.py`, `survival/`, `recommendations/` 경로 설명이 실제 `src/streaming/redis_*`, `src/models/survival_analysis.py`, `src/models/recommendations.py`와 불일치.
- [ ] MLflow 보너스를 실제 파이프라인에 연결해야 함. `MLflowTracker`/`ModelRegistry`는 있으나 `run_train`에서 tracker를 연결하지 않고 `run_all`의 `mlflow_logging`은 no-op임.
- [ ] MLflow model registry와 scoring/serving 경로를 연결해야 함. `ScoringAPI.load_model()`은 객체/path 중심이고 registry production model 로딩이 직접 연결되지 않음.
- [ ] Redis Streams producer/consumer를 실행 가능한 compose service 또는 CLI mode로 묶어야 함. 현재 producer/consumer 클래스는 있으나 별도 streaming worker/mode가 없음.
- [ ] 보너스 추천 기능이 “개인화 상품 추천” 요구와 맞지 않음. 현재는 coupon/push/email/loyalty 같은 리텐션 액션 추천 중심임.
- [ ] Survival 산출물을 대시보드 기대와 맞춰야 함. `run_survival`은 요약 JSON만 저장하고 loader는 `survival_data.csv`, `survival_curves.json`를 기대함.
- [ ] MLflow 포트 문서 불일치를 수정해야 함. README는 `localhost:5000`을 안내하지만 compose host 기본값과 deployment docs는 `5001` 계열임.
- [ ] ruff/PEP8 검증 절차를 문서/CI로 명시해야 함. `ruff` 의존성은 있으나 `pyproject.toml`/CI/README 검증 절차가 부족함.

## 우선순위 제안

1. `docker compose up` 기본 실행, Parquet 의존성, `run_all` 필수 단계, 결과 파일 생성부터 고치기.
2. `run_train` SHAP/metrics schema, 고객별 churn prediction, DL artifact 생성 경로를 고치기.
3. Uplift/CLV/Budget/A-B 산출물 스키마를 대시보드와 맞추기.
4. Monitoring report alert bug와 dashboard loader 경로를 고치기.
5. 문서/보너스/품질 절차를 실제 구현 기준으로 정리하기.

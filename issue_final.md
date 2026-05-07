# issue_1~4.md 교차검증 최종 정리

## 검증 방식

- Agent A: 실행/패키징/산출물/CLI/Docker 관점
- Agent B: 시뮬레이터/코호트/피처 엔지니어링 관점
- Agent C: ML/DL/SHAP/Uplift/CLV/Segmentation 관점
- Agent D: Budget/A-B/Dashboard/Monitoring/Docs/Quality 관점

각 에이전트는 `issue_1.md`~`issue_4.md`를 읽고 현재 저장소의 코드, 설정, 문서, 산출물을 read-only로 대조했다. 아래는 4개 관점의 교차검증을 통합한 최종 정정 사항이다.

## 전체 판정

`issue_1.md`와 `issue_2.md`는 대체로 현재 저장소 상태와 잘 맞는다. 특히 `run_all` 필수 단계 누락, SHAP API 불일치, DL pseudo-sequence, Uplift 4분면 1축 분류, CLV 운영 검증 부재, 대시보드 loader 경로/스키마 불일치, monitoring alert 구조 불일치는 유지해야 하는 지적이다.

`issue_3.md`는 종합 방향은 맞지만 일부 파트의 "완전 충족" 판정이 과하다. 특히 A/B, Dashboard, Monitoring, Docs/Quality는 구현 요소가 많더라도 end-to-end 산출물과 실제 loader 연결 기준에서는 PARTIAL로 낮춰야 한다.

`issue_4.md`는 오래된 감사 시점의 표현이 섞여 있어 정정이 가장 많이 필요하다. 특히 "results 빈 폴더", "CLV 저장 코드 없음", "cohort PNG 부재", "BG/NBD 미설치 때문에 CLV 실패" 같은 문장은 현재 코드/파일 상태 기준으로 틀리거나 과장되어 있다.

## 최종 정정 목록

### 1. `results/` 디렉터리 상태

- 틀린 주장: `results/`가 비어 있거나 디렉터리 자체가 없다는 표현.
- 정정: 현재 `results/`에는 파일이 있다. 존재 파일은 `budget_optimization.csv`, `cohort_analysis.json`, `cohort_retention_curves.png`, `cohort_retention_heatmap.png`, `cohort_retention_matrix.csv`다.
- 유지할 이슈: 핵심 제출 산출물 다수는 여전히 없다. 현재 없는 파일은 `model_metrics.json`, `shap_summary.png`, `clv_predictions.csv`, `segments_6plus.csv`, `monitoring_report.json`, `uplift_results.csv`, `qini_curve.png`, `ab_test_results.json`이다.

### 2. 코호트 산출물

- 틀린 주장: `results/cohort_retention_*.png` 산출물이 없다는 표현.
- 정정: 현재 `results/cohort_retention_curves.png`와 `results/cohort_retention_heatmap.png`는 존재한다.
- 유지할 이슈: 현재 `cohort_retention_matrix.csv`는 `cohort,0`만 있는 상태라 M1/M3/M6/M12 산출 보장으로 보기 어렵다. `run_cohort`도 churn rates, churn sequence, pre-churn event, journey funnel 결과를 저장하지 않는다.

### 3. `run_all` 파이프라인

- 유지할 이슈: `run_all`의 주석상 순서에는 `segment`, `cohort`가 포함되지만 실제 `step_handlers`에는 `run_segment`, `run_cohort`가 없다.
- 영향: `segments_6plus.csv`와 cohort 상세 산출물이 full pipeline에서 자동 생성된다고 보기 어렵다.

### 4. Parquet/Feature Store

- 틀린 주장: 파일 기반 Parquet+CSV feature store가 현재 조건에서 완전 충족된다는 표현.
- 정정: `FeatureEngineer.save_to_feature_store()`는 있지만 CLI `run_features`는 이를 호출하지 않고 `results/features.csv`만 저장한다. 또한 `to_parquet`/`read_parquet`를 쓰면서 `requirements.txt`에는 `pyarrow` 또는 `fastparquet`가 없다.
- 유지할 이슈: 전용 `src/features/store.py`는 없고 README의 feature store 구조 설명과 실제 구조가 다르다.

### 5. Simulator

- 유지할 이슈: `small_mode`는 현재 config상 500명/3개월이고 요구사항의 5,000명/6개월과 다르다.
- 유지할 이슈: simulator는 `treatment.min_group_size: 10000`을 강제하지 않는다. 단, A/B framework 쪽에서 해당 값을 읽는 코드는 있으므로 "코드 전체에서 전혀 사용 안 함"보다는 "simulator 생성 경로에서 강제하지 않음"이 정확하다.
- 유지할 이슈: `target_churn_rate`는 로드되지만 생성 결과가 15~25% 범위인지 보정/실패 처리하는 제출 파이프라인 가드는 없다.

### 6. SHAP/ML

- 유지할 이슈: `run_train`의 SHAP 호출은 현재 `ShapExplainer` API와 맞지 않는다. `ShapExplainer`는 background data를 요구하고 `save_summary_plot`, `get_top_features`를 제공하지만, `main.py`는 `ShapExplainer(ml)`, `summary_plot`, `top_features`를 호출한다.
- 틀린 주장: `Optuna` 구현.
- 정정: ML 모델은 XGBoost/LightGBM, 5-fold CV, 하드코딩된 hyperparameter grid는 있으나 Optuna 의존성/구현은 확인되지 않는다.

### 7. DL

- 유지할 이슈: production `run_train`은 `DLTrainer`의 EarlyStopping/architecture 비교 경로를 쓰지 않고 `DLChurnModel.fit()`을 직접 호출한다.
- 정정: 다만 `DLChurnModel` 자체는 config의 `dl_model.architecture`에 따라 LSTM 또는 Transformer 중 하나를 생성할 수 있다. 문제는 "아키텍처 선택 기능 전무"가 아니라 "비교/선택 trainer가 main 학습 경로에서 우회됨"이다.
- 유지할 이슈: DL 입력은 실제 이벤트 시퀀스가 아니라 tabular feature row를 반복한 pseudo-sequence다.

### 8. Uplift

- 틀린 주장: T-Learner + S-Learner가 구현되어 있으므로 요구사항을 완전 충족한다는 표현.
- 정정: 두 learner는 구현/선택 가능하지만 CLI는 `--learner` 하나만 학습한다. 요구사항의 "최소 2가지 방법을 구현하고 비교"까지는 충족하지 못한다.
- 유지할 이슈: 4분면 segmentation은 baseline churn probability 축 없이 uplift score의 부호/중앙값만 사용한다.
- 유지할 이슈: 현재 `results/uplift_results.csv`와 `results/qini_curve.png`는 없다.

### 9. CLV

- 틀린 주장: `results/clv_predictions.csv` 저장 코드가 없다는 표현.
- 정정: 파일은 현재 없지만 저장 코드는 있다. `run_clv`는 `results/clv_predictions.csv`와 `models/clv_model.pkl` 저장 코드를 포함한다.
- 틀린 주장: 상위 20% 고가치 분류가 전혀 없다는 표현.
- 정정: `CLVModel` 전용 메서드는 아니지만 `run_clv`에서 80 percentile 기준 `high_value` 플래그를 만든다.
- 틀린 주장: BG/NBD/lifetimes 미설치 자체를 CLV 실패 사유로 보는 표현.
- 정정: 요구사항은 BG/NBD+Gamma-Gamma 또는 ML 기반 중 하나를 허용한다. 실제 문제는 README/docs가 lifetimes/BG-NBD처럼 설명하는 부분이 있고, 코드는 `GradientBoostingRegressor` 기반이라는 문서-구현 불일치다.
- 유지할 이슈: `run_clv`는 같은 데이터에서 proxy target(`monetary * 12` 또는 `frequency * aov * 12`)을 학습/예측하며, 운영 산출물로 actual-vs-predicted holdout 검증 리포트를 저장하지 않는다.

### 10. Segmentation

- 틀린 주장: `segments_6plus.csv` 산출물명이 코드와 다르다는 표현.
- 정정: 산출물명은 `src/main.py`에서 `segments_6plus.csv`로 저장하도록 되어 있다.
- 유지할 이슈: 현재 파일은 없고, `run_all`에 segmentation 단계가 없다.
- 유지할 이슈: segmentation 구현은 RFM/KMeans 중심이며 요구사항의 churn probability + uplift score + CLV 결합 6+ 세그먼트와 `uplift_score * CLV` priority score를 충족하지 못한다.

### 11. Budget Optimization

- 틀린 주장: 예산 최적화가 "거의 완벽"하거나 What-if 200%까지 완료됐다는 표현.
- 정정: optimizer 내부에는 scenario/ROI 관련 함수가 있지만 CLI `run_optimize`는 단일 예산 최적화만 실행하고 `budget_optimization.csv`에는 현재 `customer_id, allocated_budget` 중심 결과만 저장한다.
- 유지할 이슈: 기본 what-if sweep은 50/100/150%이고 요구사항의 50/100/200%와 다르다.
- 유지할 이슈: CLI 산출물에 expected revenue/ROI/what-if summary가 없다.

### 12. A/B Test

- 틀린 주장: A/B 테스트 완전 충족.
- 정정: 통계 유틸과 문서는 상당히 구현되어 있으나 end-to-end 산출/대시보드 연결 기준으로는 PARTIAL이 맞다.
- 유지할 이슈: `run_ab_test`는 simulator 컬럼이 없으면 5,000/5,000 synthetic fallback을 만들고, CLI는 `results/ab_test_results.json`만 저장한다. 현재 해당 파일도 없다.
- 유지할 이슈: 대시보드 상세 A/B 화면은 `data/artifacts/ab_test_detailed.json`의 `experiments[]` 계열 스키마를 기대해 CLI 산출물과 직접 맞지 않는다.

### 13. Dashboard

- 틀린 주장: 대시보드가 완전 충족.
- 정정: 페이지/컴포넌트 수는 충분하지만 실제 pipeline output과 dashboard loader의 기본 경로/파일명/스키마가 맞지 않아 sample fallback으로 보일 수 있다.
- 유지할 이슈: loader 기본 경로는 `data/artifacts`이고, 예를 들어 budget은 `budget_results.csv`와 `allocated_budget_krw/roi`를 기대하지만 pipeline은 `results/budget_optimization.csv`와 `allocated_budget`을 만든다. CLV도 loader는 `clv_data.csv/clv_predicted`를 기대하지만 CLI는 `clv_predictions.csv/predicted_clv`를 만든다.
- 유지할 이슈: `src/dashboard/app.py`는 `monitoring_view.render_model_monitoring`을 import한 뒤 같은 이름의 로컬 함수를 다시 정의한다.

### 14. Monitoring

- 틀린 주장: monitoring alert/report와 AUC/Precision/Recall time tracking이 완전 구현되었다는 표현.
- 정정: PSI/KS detector는 있지만 `run_monitor`는 `feature_alerts`가 아니라 존재하지 않는 `.alerts` 속성을 검사한다. 따라서 CLI report의 alert 배열이 비게 될 수 있다.
- 유지할 이슈: `results/monitoring_report.json`은 현재 없다.
- 유지할 이슈: AUC/Precision/Recall 시간별 변화는 `monitoring_report.json`에 저장되지 않고, 대시보드는 sample/MLflow run 기반 표시 중심이다.

### 15. Docs/Quality/Bonus

- 틀린 주장: 문서/코드 품질 완전 충족.
- 정정: docstring과 문서량은 충분하지만 완전 충족 판정은 과하다.
- 유지할 이슈: README/docs에는 실제 없는 경로(`src/features/store.py`, `realtime_scorer.py`, `src/survival/`, `src/recommendations/`)가 섞여 있다. MLflow 포트 설명도 README의 5000 계열과 compose/deployment의 5001 계열이 섞여 있다.
- 유지할 이슈: `ruff`는 의존성에 있지만 `pyproject.toml`/ruff 설정/CI 파일은 확인되지 않는다.
- 유지할 이슈: MLflow tracker/registry, Redis producer/consumer, survival/recommendation 모듈은 있으나 `run_all`의 일부 단계는 no-op이거나 compose service/CLI worker로 완전히 묶이지 않은 부분이 있다.

## 최종 우선순위

1. `docker compose up` 기본 실행 의미를 정리하고, `SMALL=${SMALL:-true}`가 최종 제출 조건과 충돌하는지 결정한다.
2. `requirements.txt`에 Parquet 엔진을 추가하거나 feature store 저장을 CSV fallback까지 안전하게 고친다.
3. `run_all`에 `run_segment`, `run_cohort`를 실제 단계로 포함하고 필수 산출물 체크리스트를 자동 생성하도록 맞춘다.
4. `run_train`의 SHAP API 호출을 수정하고 `model_metrics.json`, `shap_summary.png`, `models/dl_churn_model.pt` 생성 경로를 검증한다.
5. Uplift 2 learner 비교, churn probability 축 기반 4분면, CLV/uplift/churn 결합 segmentation, ROI/what-if 산출물을 구현한다.
6. Dashboard loader의 기본 경로/파일명/스키마를 pipeline output과 통일한다.
7. Monitoring report가 `feature_alerts`를 읽도록 수정하고 성능 time-series 산출물을 추가한다.
8. README/docs의 실제 없는 경로, BG/NBD/lifetimes 표현, MLflow 포트, bonus 기능 연결 설명을 실제 구현 기준으로 정정한다.

## 구현 반영 기록

각 이슈가 반영된 위치와 최신 검증 증거는 `docs/requirement_traceability.md`에 기록했다.

- 필수 산출물 체크리스트: `results/required_artifacts_checklist.json` 기준 20/20 충족, missing 없음.
- 전체 테스트: `OMP_NUM_THREADS=1 LIGHTGBM_NUM_THREADS=1 /tmp/capstone-codex-py312/bin/python -m pytest -q` 기준 `2503 passed, 85 warnings`.
- Small E2E: `python src/main.py --mode all --small --quiet` 재실행 완료.

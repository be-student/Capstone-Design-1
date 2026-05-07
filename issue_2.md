# issue_2: require.md 기준 12분할 감사 결과

감사 대상: `require.md`의 고객 이탈 예측/리텐션 ROI 최적화 시스템 요구사항  
감사 방식: 명세를 12개 범위로 나누고, 무 컨텍스트 에이전트 6개씩 2라운드로 정적 감사했다. 이후 근거 품질 리뷰를 별도 수행한다.  
주의: 일부 에이전트가 감사 중 테스트를 실행하면서 `.audit/`, `results/`, `__pycache__/` 변경을 만들었다. 아래 “실제 산출물 없음” 판정은 각 에이전트가 확인한 감사 시점 기준이며, 테스트 실행 후 생성된 `results/` 파일은 최종 산출물 증거로 보지 않는다.

## 전체 결론

현재 저장소는 구현 골격과 문서가 많이 갖춰져 있지만, 최종 제출 가능 상태로 보기는 어렵다. 가장 큰 블로커는 다음이다.

1. 필수 체크리스트 중 핵심 `results/` 파일 일부가 감사 시점에 없었다.
   - 문서 산출물(`docs/feature_dictionary.md`, `docs/model_report.md`, `docs/retention_strategy.md`, `docs/ab_test_report.md`, `docs/uplift_analysis.md`)은 존재한다.
   - 다만 `results/clv_predictions.csv`, `results/segments_6plus.csv`, `results/monitoring_report.json`, `results/shap_summary.png`가 없다는 증거가 여러 파트에서 반복 확인됐다.
   - Part 1 감사 명령 출력: `MISSING data`, `MISSING results`, `MISSING results/clv_predictions.csv`, `MISSING results/segments_6plus.csv`, `MISSING results/monitoring_report.json`, `MISSING results/shap_summary.png`.

2. 명세의 직접 실행 예시가 안정적으로 입증되지 않았다.
   - `python src/main.py --help`는 `python` 명령 없음으로 실패했고, `python3 src/main.py --help`는 `ModuleNotFoundError: No module named 'numpy'`로 실패했다.
   - 예산 최적화 감사에서는 `/usr/local/bin/python3 src/main.py --mode optimize --budget 123456`가 `ModuleNotFoundError: No module named 'src'`로 실패했다.

3. 일부 핵심 요구가 문서에는 있으나 실제 파이프라인/산출물로 연결되지 않는다.
   - SHAP summary/top-10 출력 API 호출 불일치.
   - DL 모델은 고객 행동 이벤트 시퀀스가 아니라 tabular pseudo-sequence를 학습.
   - 6+ 세그먼트는 churn probability + uplift + CLV 기반이 아니라 RFM 기반.
   - CLV는 12개월 실제 holdout 검증이 아니라 proxy target 중심.
   - monitoring report는 alert key 구조 불일치 가능성이 있다.

## 우선순위 이슈

### P0: 제출을 막는 실행/산출물 문제

- 핵심 결과 산출물 생성 파이프라인을 실제로 실행하고, 요구 파일명을 맞춰 저장해야 한다.
  - 요구 파일: `results/clv_predictions.csv`, `results/segments_6plus.csv`, `results/monitoring_report.json`, `results/shap_summary.png`, uplift score CSV, Qini curve, budget optimization 결과.
  - 일부 산출물 생성 코드와 문서 파일은 존재한다. 문제는 감사 시점에 제출 체크리스트의 핵심 `results/*` 파일 일부가 없거나, 파일명이 dashboard/loader 기대값과 맞지 않는다는 점이다.

- `python src/main.py --mode ...` 형태의 직접 실행을 고쳐야 한다.
  - Part 10 근거: `/usr/local/bin/python3 src/main.py --mode optimize --budget 123456` -> `ModuleNotFoundError: No module named 'src'`.
  - Part 1 근거: `python` 명령 없음, `python3 src/main.py --help` -> `No module named 'numpy'`.
  - 실패 원인은 interpreter/env에 따라 다르므로 Docker 내부 실행과 로컬 직접 실행을 각각 명세에 맞춰 검증해야 한다.

- SHAP 출력 경로가 현재 `ShapExplainer` API와 맞지 않는다.
  - `src/models/shap_explainer.py:207-235`에는 `save_summary_plot(X, output_path)`가 있다.
  - `src/main.py:286-290`은 `ShapExplainer(ml)`로 필수 `background_data` 없이 만들고, 존재하지 않는 `summary_plot`/`top_features`를 호출한다.
  - 따라서 `run_train` 경로에서는 `results/shap_summary.png`와 top-10 feature output 생성이 사실상 실패한다.

### P1: 명세 핵심 로직 불일치

- 시뮬레이터 small mode가 요구와 다르다.
  - 요구: small mode 5,000명/6개월.
  - 실제: `config/simulator_config.yaml:19-22`는 `num_customers: 500`, `simulation_months: 3`, `simulation_days: 90`.
  - `src/main.py:895-896`의 CLI help와 문서는 5,000명/6개월을 말해 설정과 설명도 서로 불일치한다.

- Treatment/Control 각각 10,000명 이상을 보장하지 않는다.
  - `config/simulator_config.yaml:42-44`에 `min_group_size: 10000`이 있으나 코드 사용 근거가 없다.
  - `src/data/generator.py:126-140`은 Bernoulli 랜덤 분할이라 정확한 최소 인원 보장이 아니다.

- 생성 데이터 이탈률 15~25%를 강제/검증하지 않는다.
  - `target_churn_min/max`는 설정에서 읽지만, 실제 보정 또는 실패 처리 근거가 없다.
  - 관련 테스트도 5~40%로 완화되어 있다.

- DL 모델 요구의 “고객 행동 시퀀스 입력”이 약하다.
  - LSTM/Transformer 구현은 있으나 `src/models/churn_model.py:935`, `962`에서 tabular feature를 `sequence_window`만큼 복제한 pseudo-sequence를 쓴다.
  - 이벤트 타입 embedding 기반 production path는 확인되지 않았다.
  - `EarlyStopping`은 `src/models/dl_trainer.py`에 있고 해당 trainer 내부에서는 적용된다. 다만 production `run_train`이 쓰는 `DLChurnModel.fit`은 고정 epoch loop다.

- 6+ 세그먼트가 churn probability + uplift score + CLV 기반이 아니다.
  - `src/main.py:641-645`와 `src/features/segmentation.py:181-238`은 RFM 기반이다.
  - `segments_6plus.csv`의 `priority_score`는 `monetary * (1 - recency/max_recency)`이며, 요구 예시인 `uplift_score * CLV`가 아니다.

- Uplift 4분면 분류에 기본 이탈 확률 기준이 반영되지 않는다.
  - `src/models/uplift_model.py:197-242`는 uplift score 중앙값/양음수 중심으로 분류한다.
  - 요구의 “기본 이탈 확률 높음/낮음” 기준은 구현에 없다.

- CLV 산출은 12개월 예측 요구를 proxy로만 충족한다.
  - `src/main.py:358`에서 `monetary * 12` 또는 `frequency * aov * 12` proxy를 만들고 같은 데이터에 학습/예측한다.
  - 운영 파이프라인이 actual-vs-predicted 검증 리포트를 저장하는 근거가 없다.

### P2: 파이프라인 연결/검증 보강

- 코호트 분석은 함수는 있으나 CLI 저장/검증 범위가 부족하다.
  - retention matrix는 M0 포함 0~12 period를 허용하지만, M1/M3/M6/M12를 명시적으로 추출/검증하는 경로는 확인되지 않았다.
  - 가입월은 `customers.signup_date`가 아니라 첫 이벤트 월 기준이다.
  - `extract_churn_sequences`, `analyze_pre_churn_events`, `compute_journey_funnel`은 구현되어 있으나 `rg` 기준 테스트/CLI/docs 호출처가 정의부 외 확인되지 않았다.

- 피처 엔지니어링은 대부분 구현됐지만 이상치 처리와 feature store 연결이 약하다.
  - 결측/무한대 처리는 구현되어 있고 테스트도 있다. 다만 outlier 테스트는 NaN/inf 중심이며, monetary/session/ratio 등에 대한 상한 캡핑, IQR, winsorize 등 일반적 이상치 처리 근거는 부족하다.
  - `FeatureEngineer.save_to_feature_store()`는 있으나 `src/main.py:666-677`의 `run_features`는 `results/features.csv`만 직접 저장한다.

- ML threshold 분석이 파이프라인에 연결되지 않는다.
  - `src/models/churn_model.py:563-668`에 `analyze_threshold`가 있으나 호출처가 정의부 외 확인되지 않았다.

- Uplift learner 2개 비교가 자동 실행되지 않는다.
  - T-Learner와 S-Learner 구현은 있으나 `src/main.py:319-327`은 선택된 learner 하나만 학습/평가한다.
  - simulator 산출물에는 treatment/control이 있지만, `run_uplift`는 `treatment_group` 컬럼이 없으면 전부 treatment로 대체하므로 실행 입력 검증도 필요하다.

- 예산 최적화 what-if 기본 시나리오가 요구와 다르다.
  - 요구: 50%, 100%, 200%.
  - `src/optimization/budget_optimizer.py:1210` 기본 sweep은 50%, 100%, 150%.
  - 직접 `budget_levels=[B*0.5, B, B*2.0]`로 호출할 수는 있으므로, 문제는 기본값/CLI 산출 경로가 요구 시나리오를 자동 생성하지 않는 점이다.

- 예산 최적화 CLI 산출물에 ROI/what-if가 빠져 있다.
  - `src/models/budget_optimizer.py:421-443`에는 ROI 계산이 있으나, `src/main.py:447-449`의 CLI 결과는 `budget_optimization.csv`, `total_budget`, `allocated` 중심이다.

- A/B test는 통계 로직은 좋지만 교란/균형 진단과 실제 결과 파일이 부족하다.
  - Power, z-test/chi-square, p-value, CI는 구현되어 있다.
  - 공변량 balance table, persona별 balance, confounding 진단 구현 근거는 없다. 이는 A/B 기능 요구의 핵심 항목보다는 `require.md` 학습 목표의 “교란 통제 전략 설명”을 보강해야 하는 이슈다.

- 모니터링 report는 성능 시간 추적과 저장 구조가 약하다.
  - PSI/KS 탐지는 구현되어 있다.
  - `monitoring_report.json` 저장 경로는 `src/main.py:713-731`에 있다. 다만 저장 내용은 PSI/KS 중심이고, AUC/Precision/Recall 시간별 저장은 대시보드 샘플/MLflow history 표시 중심이다.
  - `src/main.py:716-724`는 `psi_report.alerts`, `ks_report.alerts`를 찾지만 실제 report 객체는 `feature_alerts` 구조로 보인다.

## 12개 범위별 판정

| # | 범위 | 판정 | 핵심 근거 |
|---|---|---|---|
| 1 | 최종 산출물/패키징/CLI/Docker | PARTIAL | `docker compose config`는 OK이나 `docker compose up` 런타임 성공은 미검증. CLI mode는 등록되어 있으나 직접 실행 검증 실패. 핵심 `results/` 산출물 일부 없음. |
| 2 | 고객 행동 시뮬레이터 | PARTIAL | 6 persona와 8 event는 PASS. small mode, treatment/control 최소 인원, 이탈률 target 강제는 미흡. |
| 3 | 코호트/여정 분석 | PARTIAL | retention matrix/plot 저장 로직은 있으나 M1/M3/M6/M12 명시, signup month 기준, churn pattern/funnel 저장 연결 부족. |
| 4 | 피처 엔지니어링 | PARTIAL | RFM, 30+ dictionary, 변화율, session, sequence, time, journey features는 대체로 PASS. 이상치 처리와 feature store 연결은 PARTIAL. |
| 5 | ML 이탈 예측 | PARTIAL | XGBoost/LightGBM, 5-fold CV, tuning은 PASS. SHAP CLI 연결, top-10 output, threshold 산출물, 실제 AUC 산출물은 미흡. |
| 6 | DL 이탈 예측 | PARTIAL | PyTorch LSTM/Transformer는 있으나 실제 고객 행동 시퀀스 입력과 early stopping production path, DL artifact/log가 부족. |
| 7 | Uplift Modeling | PARTIAL | Treatment/control 사용과 uplift score 산출은 PASS. 2 learner 비교, churn-risk 기반 4분면, 실제 uplift results/Qini artifact는 부족. |
| 8 | CLV 예측 | PARTIAL | ML 기반 CLV 모델은 있음. 12개월 proxy, actual-vs-predicted report, `results/clv_predictions.csv`와 Top-N/분포 리포트는 부족. |
| 9 | 6+ 세그먼트/우선순위 | FAIL/PARTIAL | 구현은 RFM 8세그먼트. 요구의 churn + uplift + CLV 기반 분류와 `uplift * CLV` priority가 아님. |
| 10 | 리텐션 전략/예산 최적화 | PARTIAL | 전략 문서/LP 구현/관련 테스트는 좋음. 직접 CLI 실패, ROI/what-if CLI 산출물 부족, 200% scenario 불일치. |
| 11 | A/B 테스트 | PARTIAL | power, z-test/chi-square, p-value, CI, report는 PASS. 현재 result file이 없고, 학습 목표 관점의 balance/confounding 설명 근거가 부족. |
| 12 | 대시보드/모니터링/문서/품질 | PARTIAL | dashboard pages/refresh/PSI/KS/docs는 PASS. monitoring report 산출물과 성능 time-series 저장, docstring/config 분리는 부분 충족. |

## 세부 근거

### 1. 최종 산출물/패키징/CLI/Docker

- PASS: Git remote 존재. `git remote -v` -> `origin https://github.com/be-student/Capstone-Design-1.git`.
- PASS: 현재 branch는 main. `git rev-parse --abbrev-ref HEAD` -> `main`.
- UNKNOWN: repository public 여부는 로컬만으로 검증 불가.
- PARTIAL: `docker-compose.yml:28-180`에 `mlflow`, `redis`, `pipeline`, `dashboard`가 있고 `docker compose config --quiet`는 OK. 다만 `docker compose up` 런타임 성공과 서비스 health는 검증되지 않았다.
- PARTIAL: `docker-compose.yml:110-112`의 `SMALL=${SMALL:-true}`는 최종 20K 기본 실행 기대와 충돌 가능성이 있다.
- PARTIAL/FAIL: CLI mode는 `src/main.py:816-821`, `829`, `883-885`에 있으나 로컬 직접 실행은 dependency/import 문제로 실패했다.
- FAIL: 감사 시점 핵심 `results/*` 산출물 일부가 없었다. 문서 산출물은 존재한다.

### 2. 고객 행동 시뮬레이터

- PASS: 6 persona는 `config/simulator_config.yaml:241-426`에 정의되어 있다.
- PASS: 8 event type은 `config/simulator_config.yaml:200-210`, 생성 로직은 `src/data/generator.py:326-407`.
- PARTIAL: 행동 변화는 방문/검색/장바구니/구매 확률 감소 중심으로 구현되어 있고 `session_time_decay`가 일부 로직에 간접 사용된다. 다만 실제 `session_duration` 값은 이벤트 스키마에서 저장되지 않는다.
- PARTIAL: marketing response 설정과 처리 효과는 있으나 coupon 발송/push notification intervention event 또는 conversion/no-response/backfire label은 없다.
- PARTIAL: 기본 20,000명/12개월 설정은 있으나 실제 생성 결과 파일은 감사 시점 확인되지 않았다.
- FAIL: small mode 설정은 요구 5,000명/6개월이 아니라 500명/3개월이며, CLI help/문서와 설정도 불일치한다.
- PARTIAL: treatment/control 최소 10,000명은 설정만 있고 구현 보장 없음.
- PARTIAL: churn rate 15~25% target은 설정만 있고 강제/검증 근거 부족.

### 3. 코호트/고객 여정 분석

- PARTIAL: `CohortAnalyzer`는 M0 포함 0~12 period retention matrix를 계산할 수 있지만 M1/M3/M6/M12를 명시적으로 추출하지 않는다.
- PARTIAL: cohort 기준이 `customers.signup_date`가 아니라 첫 이벤트 월이다.
- PARTIAL: CLI는 retention matrix/heatmap/curve 저장 로직이 있으나 churn rate comparison 저장/시각화는 부족하다.
- PARTIAL: churn sequence top-5, pre-churn event frequency, journey funnel 함수는 있으나 `rg` 기준 CLI/test/docs 호출처가 정의부 외 확인되지 않았다.

### 4. 피처 엔지니어링

- PASS: RFM 구현은 `src/features/feature_engineering.py:125-186`.
- PASS: `docs/feature_dictionary.md`는 33개 feature 정의를 포함한다.
- PASS: 행동 변화율 7개(`src/features/feature_engineering.py:203`), session quality 5개(`src/features/feature_engineering.py:423`), sequence feature 4개(`src/features/feature_engineering.py:518`), time/weekend feature(`src/features/feature_engineering.py:612`), journey stage feature(`src/features/feature_engineering.py:703`)가 구현되어 있다.
- PARTIAL: 결측/무한대 처리와 안전 나눗셈은 있으나 outlier 테스트는 NaN/inf 중심이고, 체계적 이상치 처리 근거가 부족하다.
- PARTIAL: file-based feature store API는 있으나 `run_features`가 이를 호출하지 않는다.

### 5. ML 기반 이탈 예측

- PASS: `src/models/churn_model.py`는 XGBoost와 LightGBM을 구현하고 5-fold CV/tuning을 수행한다.
- PARTIAL: class imbalance는 최종 학습에는 반영되나 CV/tuning 단계에는 동일 적용 근거가 약하다.
- PARTIAL: AUC >= 0.78은 테스트 코드 assert는 있으나 실제 `results/` 산출물로 확인되지 않았다.
- FAIL: SHAP summary plot/top-10 output은 `src/main.py`와 `ShapExplainer` API 불일치로 `run_train` 경로에서 사실상 실패한다.
- PARTIAL: threshold precision-recall tradeoff 함수는 있지만 pipeline 호출/산출물 연결이 없다.

### 6. DL 기반 이탈 예측

- PARTIAL: LSTM/Transformer PyTorch 모델은 있다.
- PARTIAL: production train path는 행동 event sequence가 아니라 tabular pseudo-sequence를 사용한다.
- PARTIAL: sequence preprocessing/padding utility는 있으나 `run_train` path에서 쓰이지 않는다.
- PARTIAL: EarlyStopping class와 적용 trainer는 있으나, production `run_train`이 호출하는 `DLChurnModel.fit`은 고정 epoch loop다.
- PARTIAL: ML/DL/ensemble same-test-set 비교 및 `results/model_metrics.json` 저장 코드는 있으나 실제 파일은 감사 시점 없음.
- FAIL: `models/dl_churn_model.pt`, DL training log, 실제 실행 결과 기반 비교 report 산출물이 없다. `docs/model_report.md`는 존재하나 정적/예상 성능 문서 중심이다.

### 7. Uplift Modeling

- PASS/PARTIAL: simulator 산출물에 `treatment_group`이 있으면 treatment/control data를 학습 입력으로 사용한다. 다만 `run_uplift`는 컬럼이 없으면 전부 treatment로 대체한다.
- PARTIAL: T-Learner와 S-Learner 구현은 있으나 같은 실행에서 두 방법 비교 결과를 만들지 않는다.
- PASS: customer-level uplift score 산출 로직은 있다.
- PARTIAL: 4분면 세그먼트는 `segment_customers(self, uplift_scores)`가 uplift median/부호 중심으로 분류하며, 요구의 기본 churn probability high/low 기준을 반영하지 않는다.
- PARTIAL/FAIL: Qini curve와 uplift CSV 저장 로직은 있으나 실제 `results/qini_curve.png`, `results/uplift_results.csv`는 감사 시점 없음.
- PARTIAL: Persuadables 특성 분석은 문서 중심이며 데이터 집계 산출물 근거가 부족하다.

### 8. CLV 예측

- PASS: `src/models/clv_model.py`는 ML 기반 `GradientBoostingRegressor` CLV 모델을 구현한다.
- PARTIAL: 12개월 CLV는 proxy target 기반이며 실제 미래 12개월 holdout 검증이 아니다.
- PARTIAL: top 20% high-value flag는 코드에 있으나 별도 distribution report artifact는 없다.
- PARTIAL: actual-vs-predicted validation은 synthetic target 기반 테스트 근거만 있고 운영 artifact가 없다.
- FAIL: `results/clv_predictions.csv`와 Top-N/분포 리포트 파일이 감사 시점 존재하지 않는다.
- PARTIAL: dashboard loader는 `clv_data.csv`를 찾고 없으면 sample CLV를 생성한다. 명세/파이프라인의 `results/clv_predictions.csv` 부재가 dashboard fallback에 가려질 수 있다.

### 9. 고객 세그먼테이션/우선순위

- PARTIAL: 8개 RFM 세그먼트는 구현되어 있다.
- FAIL: 요구의 churn probability + uplift score + CLV 기반 6+ 세그먼트가 아니다.
- PARTIAL: segment summary는 count/percentage/RFM 평균과 optional churn probability만 있고 avg CLV는 요약 함수에 없다.
- FAIL: priority score가 `uplift_score * CLV` 계열이 아니라 RFM monetary/recency 기반이다.
- PASS: dashboard segmentation 시각화는 있다.
- FAIL: `results/segments_6plus.csv` 실제 산출물은 감사 시점 없음.

### 10. 리텐션 전략/예산 최적화

- PASS: `docs/retention_strategy.md`에 6개 이상 전략, 비용/효과, 목적함수/제약조건 수식이 있다.
- PASS: LP optimizer 구현과 관련 테스트가 있다.
- PARTIAL: 구현 목적함수는 명세 수식에 `churn_prob`를 추가한다.
- PARTIAL: what-if 기본 scenario가 50/100/150%라 200% 요구와 불일치한다. 직접 budget levels를 넘길 수는 있으나 기본값/CLI 산출 경로가 50/100/200%를 자동 생성하지 않는다.
- PARTIAL: ROI 계산 API는 있으나 CLI 산출물에는 ROI/what-if 저장이 부족하다.
- FAIL: `/usr/local/bin/python3 src/main.py --mode optimize --budget 123456` 직접 실행이 `ModuleNotFoundError: No module named 'src'`로 실패했다. 기본 `python3`에서는 `numpy` 미설치에 먼저 막힌다.
- PASS: `/usr/local/bin/python3 -m pytest -q tests/test_budget_optimization.py tests/test_lp_budget_optimizer.py tests/test_budget_cost_config.py tests/test_main_cli.py tests/test_cli_entrypoint.py`는 `308 passed, 2 warnings in 29.56s`로 통과했다.

### 11. A/B 테스트 설계/분석

- PASS: simulator `treatment_group`/`churn_label` 기반 A/B path가 있다.
- PASS: power/sample size, z-test/chi-square, p-value, 95% CI, `p < 0.05` significance 판단이 구현되어 있다.
- PASS: `docs/ab_test_report.md`가 Power, p-value, CI, 해석을 포함한다.
- PARTIAL: random assignment와 exact treatment count utility는 있으나 covariate balance/confounding 진단 구현 근거가 없다. 이는 기능 요구보다는 학습 목표/설명 가능성 보강 이슈다.
- PARTIAL: dashboard A/B page는 `results/ab_test_results.json`이 없으면 fallback sample data를 표시한다.
- FAIL: `run_ab_test` 저장 코드는 있으나 실제 A/B result file은 감사 시점 확인되지 않았다.

### 12. 대시보드/모니터링/문서/코드 품질

- PASS: dashboard pages, routing, refresh button, customer priority list가 있다.
- PASS: 이탈 분포, segment, budget/ROI, A/B, CLV, Uplift, cohort, monitoring 시각화 범위가 있다.
- PASS: PSI와 KS/Chi-square drift detection, alert callback 로직이 있다.
- PARTIAL: `monitoring_report.json` 저장 경로는 있으나 report 내용은 PSI/KS 중심이며, AUC/Precision/Recall 시간별 tracking은 dashboard/MLflow history 표시 중심이다.
- FAIL: `results/monitoring_report.json` 실제 산출물은 감사 시점 없음.
- PASS: README architecture/data flow diagram과 주요 docs가 있다.
- PARTIAL: AST 집계 기준 functions 513개 중 docstring missing 19개.
- PARTIAL: config 분리와 secret 관리는 대체로 괜찮으나 localhost/port/sqlite fallback hardcoding이 남아 있다.

## 권장 수정 순서

1. `python src/main.py --mode ...` 직접 실행 import 문제를 먼저 해결한다.
2. small mode를 명세대로 5,000명/6개월로 맞추고, treatment/control 최소 인원 및 churn rate 15~25% 검증을 강제한다.
3. `run_train`의 SHAP 호출을 `ShapExplainer` 실제 API에 맞추고 `results/shap_summary.png`, top-10 importance를 저장한다.
4. `run_all` 또는 docker pipeline이 필수 산출물을 한 번에 생성하도록 연결한다.
5. segmentation을 churn probability + uplift score + CLV 기반으로 다시 만들고 `priority_score = uplift_score * CLV` 계열로 저장한다.
6. DL training path를 실제 event sequence preprocessing/padding/embedding 기반으로 연결하고 early stopping과 로그/모델 artifact를 저장한다.
7. CLV actual-vs-predicted validation artifact와 `results/clv_predictions.csv`/Top-N/분포 리포트를 만든다.
8. Uplift learner 2개 비교, churn-risk 기반 4분면, Qini artifact를 CLI 결과로 저장한다.
9. budget CLI에 ROI와 50/100/200% what-if 결과 저장을 추가한다.
10. monitoring report에 PSI/KS threshold 초과와 AUC/Precision/Recall time series를 실제 구조로 저장한다.
11. A/B balance/confounding 진단 표를 생성하고 report/dashboard에 연결한다.
12. 결과 파일이 없는 상태에서 dashboard가 sample fallback으로 통과해 보이지 않도록, 제출 검증 모드에서는 missing artifact를 명확히 실패 처리한다.

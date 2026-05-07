# 캡스톤 PRD vs 실제 코드 구현 현황

> 13개 PRD 파트 1차 검증 + 5개 그룹 spot-check 재검증 (총 50개 claim, 본질 정확률 100%)

> 저장소: `C:\Users\yoonc\Capstone-Design-1` · 검증일: 2026-05-02

## 종합 요약

| 파트               | 충족  | 부분/결함 | 미구현 | 종합             |
| ------------------ | ----- | --------- | ------ | ---------------- |
| P1 시뮬레이터      | 5/8   | 3/8       | 0      | 🟢 양호          |
| P2 코호트·여정     | 0/6   | 1/6       | 5/6    | 🔴 심각          |
| P3 피처 엔지니어링 | 10/10 | 0         | 0      | ✅ 완전          |
| P4 ML 이탈 예측    | 4/9   | 3/9       | 2/9    | 🟡 통합 결함     |
| P5 DL 이탈 예측    | 1/6   | 5/6       | 0      | 🔴 시퀀스 우회   |
| P6 Uplift Modeling | 4/6   | 1/6       | 1/6    | 🟡 4분면 1축     |
| P7 CLV 예측        | 0/5   | 1/5       | 4/5    | 🔴 BG/NBD 미설치 |
| P8 세그먼테이션    | 1/4   | 3/4       | 0      | 🔴 3축 결합 부재 |
| P9 예산 최적화     | 5/6   | 1/6       | 0      | 🟢 What-if 200%  |
| P10 A/B 테스트     | 6/6   | 0         | 0      | ✅ 과스펙        |
| P11 통합 대시보드  | 8/9   | 1/9       | 0      | ✅ 16페이지      |
| P12 모델 모니터링  | 3/4   | 1/4       | 0      | 🟢 산출물 부재   |
| P13 문서·코드 품질 | 8/8   | 0         | 0      | ✅ 완전          |

**핵심 결론**: 코드는 작성되어 있으나 파이프라인이 한 번도 end-to-end 실행되지 않아 `results/` 디렉토리가 빈 폴더. 다수 함수가 `main.py`에 미연결된 dead code 상태.

## 🟢 P1. 고객 행동 시뮬레이터

### ✅ 구현됨

- 6개 페르소나 정의 (`config/simulator_config.yaml:241-426`)
- 8개 이벤트 유형 생성 (`generator.py:326-407`)
- 시간 경과 행동 변화 3축 decay (visit/session/purchase_cycle)
- 마케팅 개입 페르소나별 차등 반응 (coupon_lift, push_lift, adverse_effect_prob)
- 이탈 정의 config 변경 가능 (OR/AND, 30/60일)

### ⚠ 부분 구현·결함

- small_mode가 yaml에 500명/3개월 (주석엔 5,000명/6개월로 표기 — 본문과 불일치)
- min_group_size=10000 검증 로직이 generator/orchestrator에 없음
- target_churn_min/max가 dead code (`generator.py:66-67`에서 로드만 됨)

## 🔴 P2. 코호트 및 고객 여정 분석

### ⚠ 부분 구현

- M0~M12 리텐션 매트릭스 산출 (M1/M3/M6/M12 명시 컬럼 추출은 없음)

### ❌ 미구현 (dead code)

- `extract_churn_sequences` (top 5 패턴) — 호출 0건 (`cohort_analysis.py:906`)
- `analyze_pre_churn_events` (이탈 직전 빈도) — 호출 0건 (`cohort_analysis.py:961`)
- `compute_journey_funnel` (5단계 퍼널) — 호출 0건 (`cohort_analysis.py:1035`)
- `compute_churn_rates` 시각화 — `run_cohort`에서 미호출
- `results/cohort_retention_*.png` — 빈 폴더, 산출물 부재

## ✅ P3. 피처 엔지니어링 — 완전 충족 (10/10)

| 카테고리       | PRD 요구 | 실제 구현 |
| -------------- | -------- | --------- |
| RFM            | 3개      | 5개       |
| 행동 변화율    | ≥5개     | 7개       |
| 구매 주기 이상 | 1개      | 3개       |
| 세션 품질      | ≥3개     | 5개       |
| 시퀀스         | ≥2개     | 4개       |
| 시간대별       | 1개      | 6개       |
| 여정 단계      | 1개      | 2개       |
| **합계**       | **30+**  | **33개**  |

- `feature_dictionary.md`에 33개 모두 정의
- 결측·이상치 처리: fillna(0), inf 치환, recency clip
- 피처 스토어: 파일 기반 Parquet+CSV (PRD "OR" 조건 충족)

## 🟡 P4. ML 이탈 예측 (4/9 ✅)

### ✅ 구현됨

- XGBoost + LightGBM 두 모델 학습·비교
- 5-Fold Stratified CV
- SHAP 전역 + 개별 해석 (`shap_explainer.py`)
- Top-10 피처 추출 함수 (`get_top_features(k=10)`)

### ⚠ 부분 구현

- 클래스 불균형: SMOTE 미사용 → `is_unbalance` / `scale_pos_weight` 대체
- `analyze_threshold` 함수 정의됨, 그러나 `main.py`에서 호출 0건
- 하이퍼파라미터: Optuna 미설치, 하드코딩 3-config 그리드

### ❌ 미구현·치명적 버그

- `results/shap_summary.png` 미생성 — `main.py:289-290` 메서드명 버그 (`summary_plot` → 실제 `save_summary_plot`, `top_features` → 실제 `get_top_features`). try/except로 silent fail
- AUC ≥ 0.78 실측 부재 (`model_metrics.json` 없음)

## 🔴 P5. DL 이탈 예측 — 시퀀스/EarlyStopping 우회

### ✅ 구현됨

- ML vs DL 동일 테스트셋 비교 (time_based_split 공유)

### ⚠ 부분 구현 (모두 결함)

- LSTM/Transformer PyTorch 클래스 정의됨 (`churn_model.py:675, 781`)
- `sequence_utils.create_sequences` 정의만, **`main.py` import 0건**
- EarlyStopping 클래스 존재, **main이 `DLChurnModel.fit()` 직접 호출로 우회**
- 앵상블: `EnsembleChurnModel` 가중평균 0.6/0.4 고정 — Stacking 아님
- DL 입력: pseudo-sequence (단일 시점 피처를 tile + 시간 스케일링)

### ❌ 미구현

- `models/dl_churn_model.pt` 부재

## 🟡 P6. Uplift Modeling — 4분면이 1축 분류

### ✅ 구현됨

- Treatment/Control 활용 (`fit(X, treatment, y)`)
- T-Learner + S-Learner 두 메타러너 (PRD "택2" 충족)
- Uplift Score(CATE) 산출
- Persuadables 특성 분석 문서 (`uplift_analysis.md` §8)

### ⚠ 부분 구현

- 4분면 분류: **uplift 단일축 median**으로만 분류 (PRD 요구는 Uplift × Churn Probability 2축)

### ❌ 미구현

- Qini Curve PNG: `plot_qini_curve` 함수는 있으나 `results/` 빈 폴더 → 산출물 없음

## 🔴 P7. CLV 예측 — BG/NBD 미설치

### ⚠ 부분 구현

- ML 기반 CLV: `GradientBoostingRegressor` 단일 (docs는 BG/NBD 약속, 코드와 모순)

### ❌ 미구현

- `lifetimes` 패키지 미설치 (`lifelines`만 있음, 다른 용도)
- 12개월 horizon 파라미터 부재
- 상위 20% 고가치 분류 메서드 없음
- 정확도 검증 (RMSE/MAE/R²) 메서드 없음
- `results/clv_predictions.csv` 저장 코드 없음 + 파일 부재
- 분포 리포트·Top-N 출력 없음

> docs(BG/NBD) ↔ modules(GB inspired) ↔ 코드(GB) **3자 불일치** — 보고서 거짓 진술 리스크

## 🔴 P8. 세그먼테이션 — 3축 결합 분류기 부재

### ✅ 구현됨

- 세그먼트 분류 시각화: `render_segmentation` 다중 차트

### ⚠ 부분 구현

- 6+ 세그먼트: 8개 정의됐으나 **RFM 단일축**만 사용 (PRD 요구는 이탈확률·Uplift·CLV 3축)
- 세그먼트별 통계: `avg_clv` 컬럼 누락 (avg_churn_probability는 있음)
- 우선순위 점수: `main.py:650`은 `monetary × (1 - recency/max_r)` — Uplift × CLV 아님

### ❌ 미구현

- `results/segments_6plus.csv`: 저장 코드 있으나 파일 부재

## 🟢 P9. 예산 최적화 — 거의 완벽

### ✅ 구현됨

- 세그먼트별 전략 정량 명시 (`retention_strategy.md`: 70K/30K/1K KRW, 15-25%p)
- LP 정식화: scipy `linprog(method="highs")`
- 목적함수: `Σ(uplift × clv × churn_prob × Action)` (PRD + churn_prob)
- 베이스라인 LP + proportional + uniform fallback
- ROI 산출 + NPV 옵션

### ⚠ 부분 구현

- What-if default `[0.5, 1.0, 1.5]` — **PRD 200%(2.0) 누락, 150% 대체** (`src/optimization/budget_optimizer.py:1213-1217`)

> Note: `models/budget_optimizer.py`와 `optimization/budget_optimizer.py` 두 모듈 이중화

## ✅ P10. A/B 테스트 — 완전 충족 + 과스펙 (6/6)

### ✅ 모두 구현됨

- Power Analysis: required_sample_size, compute_power, MDE 역산
- 검정: Welch t-test, Chi-square, Mann-Whitney U, Z-test (자동 선택)
- 95% CI + p-value (모든 검정)
- `is_significant` 필드 (p < 0.05)
- `docs/ab_test_report.md` 312줄 8섹션

### 🎁 PRD 요구 외 보너스

- 다중비교 보정 3종 (Bonferroni, BH-FDR, Holm)
- 순차검정 alpha-spending (O'Brien-Fleming, Pocock, Linear)
- A/B/n 다변량, 결정론적 해싱

## ✅ P11. 통합 대시보드 — 16페이지 과스펙 (8/9)

### ✅ 16개 페이지 모두 구현

1. Overview (이탈 위험 분포)
2. Churn Analytics
3. Model Performance
4. Customer Segmentation
5. Cohort Analysis (리텐션 곡선)
6. Budget Optimization (예산 + ROI)
7. A/B Testing (결과 요약)
8. Survival Analysis (보너스)
9. Model Monitoring (보너스)
10. Recommendations (우선순위 고객)
11. CLV Prediction
12. Uplift Modeling (4분면)
13. CLV & Retention Campaign
14. Real-Time Scoring (보너스)
15. MLflow Experiments (보너스)
16. System Health (보너스)

- 수동 새로고침: 🔄 버튼 → `cache_data.clear() + rerun()`
- Docker: 8501 포트 + `/_stcore/health`

### ⚠ 부분 구현

- 6세그먼트 라벨링 시각화 부재 (config는 페르소나 6개 + 세그먼트 8개)

## 🟢 P12. 모델 모니터링 (3/4)

### ✅ 구현됨

- PSI 드리프트 (GREEN<0.10/YELLOW<0.25/RED 3단계)
- KS-test (`scipy.stats.ks_2samp`) + Chi-Square
- 알림: `AlertLevel` enum + `register_alert_callback`
- AUC/Precision/Recall 시계열 추적

### ⚠ 부분 구현

- `results/monitoring_report.json`: 저장 코드 `main.py:731` 존재, run_monitor 미실행으로 파일 부재

## ✅ P13. 문서화 및 코드 품질 (8/8)

- README ASCII 아키텍처 다이어그램
- docs/ 4개 필수 문서 모두 존재
- 세그먼트 전략 제안서 (`retention_strategy.md`)
- A/B 결과 해석 리포트 (312줄)
- 5개 핵심 모듈 docstring 평균 ~95%
- src/ 9개 도메인 모듈 분리

## 즉시 보완 Top 7

-

1. SHAP 메서드명 버그 수정: [main.py](http://main.py) 289-290줄

-

1. run_cohort에 4개 dead code 호출 추가: [main.py](http://main.py) 602-631줄

-

1. CLV에 12개월 horizon, Top20%, RMSE, CSV 저장 추가

-

1. 3축 결합 6세그먼트 분류기 신설

-

1. Uplift segment_customers를 2축으로 변경

-

1. What-if default 0.5/1.0/2.0으로 수정

-

1. End-to-end 1회 실행으로 results 산출물 일괄 생성

## 검증 메타데이터

- 1차: 13개 PRD 파트별 1개씩 = 13개 서브에이전트 병렬
- 2차 spot-check: 5개 그룹(P1-P3, P4-P5, P6-P8, P9-P10, P11-P13) 50개 claim 재검증
- 결과: 50/50 본질 정확 (3건 미세 보정)

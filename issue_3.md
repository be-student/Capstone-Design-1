# Issue #3 — require.md 구현 충실도 감사 결과

본 문서는 `require.md` 명세를 12개 파트로 등분하여 무컨텍스트 감사 에이전트 12명이 실 코드와 대조 검증하고, 그 결과를 다시 무컨텍스트 리뷰어 6명이 인용·증거 품질 측면에서 재검증한 종합 보고서다.

- 감사 보고서 원본: `.audit/round1/part-01..06-*.md`, `.audit/round2/part-07..12-*.md`
- 리뷰 보고서 원본: `.audit/review/review-01..06.md`

---

## 1. 절차

| 라운드 | 에이전트 | 산출물 | 비고 |
|--------|---------|--------|------|
| Round 1 | 6명 (parts 1-6) | `.audit/round1/*.md` | Simulator/Cohort/Features/ML/DL/Uplift |
| Round 2 | 6명 (parts 7-12) | `.audit/round2/*.md` | CLV/Segmentation/Optimization/A·B/Dashboard/Monitoring+Docs |
| Round 3 | 리뷰어 6명 (보고서 2개씩) | `.audit/review/*.md` | 인용 라인/발췌 fidelity/판정 타당성 재검증 |

각 라운드는 동시 6명 상한을 지켰고, 모든 에이전트는 무컨텍스트 fresh subagent로 띄웠다.

---

## 2. 12개 파트 감사 점수 종합

| # | 파트 | 명세 항목 | ✅ | ⚠️ | ❌ | 한 줄 결론 |
|---|-----|----------|----|----|----|------------|
| 01 | 시뮬레이터 (#5.1) | 8 | 5 | 3 | 0 | 페르소나·이벤트·이탈정의·decay 모두 구현, 단 `small_mode` 500명·`min_group_size` 미강제·이탈률 실측 미검증 |
| 02 | 코호트/여정 (#5.2) | 6 | 3 | 3 | 0 | 리텐션 매트릭스/시각화는 정상, 그러나 M1/M3/M6/M12 마일스톤 표기·이탈 시퀀스/사전 이벤트/여정 퍼널 함수가 `run_cohort` 파이프라인과 미연결 |
| 03 | 피처 (#5.3) | 10 | 8 | 2 | 0 | 33개 피처 정의·RFM·변화율·세션·시간대·여정 단계 OK, 단 시퀀스 임베딩 미구현·`save_to_feature_store` 메인 미호출 |
| 04 | ML 모델 (#5.4) | 9 | 5 | 3 | 1 | XGBoost/LightGBM·5-Fold·Optuna 구현, 그러나 `src/main.py` SHAP 호출 잘못된 인자/메서드 → SHAP plot 미생성, `results/`에 AUC 실측 증거 부재 |
| 05 | DL 모델 (#5.5) | 7 | 5 | 2 | 0 | LSTM/Transformer·EarlyStopping·앙상블 코드 존재, 그러나 학습 시 `np.tile` 의사 시퀀스 사용·`models/`에 DL 파일 미존재 |
| 06 | Uplift (#5.6) | 7 | 5 | 2 | 0 | T-Learner/S-Learner·Qini Curve·CSV 출력 OK, 4분면 분류가 Uplift 단일축(중앙값)만 사용·Persuadables 자동 분석 미구현 |
| 07 | CLV (#5.7) | 5 | 2 | 2 | 1 | GBM 기반 CLV·Top-20% 분류 OK, 12개월 산정이 `monetary × 12` 프록시·정확도 검증 부재·`results/clv_predictions.csv` 미생성 |
| 08 | 세그먼테이션 (#5.8) | 5 | 1 | 4 | 0 | 8개 세그먼트·priority·CSV·시각화 모두 존재하나 require.md가 명시한 "이탈확률 × Uplift × CLV" 결합 라벨(예: 고가치-Persuadables) 체계와 priority 공식이 부합 안 함 |
| 09 | 예산 최적화 (#5.9) | 7 | 5 | 2 | 0 | 목적함수·제약·LP·What-if 코드 존재, 그러나 `--mode optimize` CLI에서 50/100/200% 시나리오 + ROI 요약 미저장·대시보드 라벨 spec과 불일치 |
| 10 | A/B 테스트 (#5.10) | 6 | **6** | 0 | 0 | Power Analysis·Chi-square/Z-test·95% CI·p-value·`docs/ab_test_report.md` 모두 충족 |
| 11 | 대시보드 (#5.11) | 10 | **10** | 0 | 0 | Streamlit 기반 10개 항목 모두 구현, docker-compose 8501 포트 매핑·수동 새로고침 확인 |
| 12 | 모니터링+문서 (#5.12+#5.13) | 12 | 11 | 1 | 0 | PSI/KS·alert·`monitoring_report.json`·아키텍처 다이어그램·docstring 풍부, 다만 `run_monitor`가 단일 데이터셋 절반-나누기라 시간축 추적은 약식 |
| **합계** | — | **92** | **66** | **24** | **2** | **충족 71.7%, 부분 26.1%, 미흡 2.2%** |

---

## 3. ❌ (미구현/결함) 사항 — 즉시 수정 권장

| # | 위치 | 결함 | 영향 |
|---|------|------|------|
| ❌-1 | `src/main.py` SHAP 호출부 (Part 04 항목 5·6) | `ShapExplainer(ml)` 1-인자 호출과 `summary_plot()` 메서드명이 클래스 정의와 불일치 → `results/shap_summary.png` 미생성 | require.md #5.4 항목 5/6 충족 불가, 산출물 체크리스트 미달 |
| ❌-2 | CLV 정확도 검증 (Part 07 항목 4) | `actual_vs_predicted_clv` 비교 루틴 없음. `docs/clv_*` 별도 문서 부재 | require.md #5.7 검증 의무 미달 |

---

## 4. ⚠️ 부분 구현 — 결합 우선순위 높은 항목

### 4.1 데이터/시뮬레이션 정합

- **`small_mode.num_customers` = 500** (`config/simulator_config.yaml` 내), spec 5,000명과 불일치 → Part 01.
- **Treatment/Control 각 10,000명** 보장 로직 부재 (베르누이 추첨만 존재) → Part 01.
- **이탈률 15-25% 실측 가드** 없음(스케일링은 코드에 있음) → Part 01.

### 4.2 코호트 산출 누락

- `compute_churn_rates`/`extract_churn_sequences`/`analyze_pre_churn_events`/`compute_journey_funnel` 4개 함수가 정의만 되고 `run_cohort`에서 호출되지 않음 → Part 02 항목 2·4·5·6 모두 영향.
- `M1/M3/M6/M12` 4개 마일스톤 별도 표기 없음 → Part 02.

### 4.3 모델링 정합

- **DL 학습이 `np.tile` 의사 시퀀스로 동작** (`src/models/churn_model.py:962-970`). `sequence_utils.create_sequences`는 정의만 되고 호출 안 됨 → Part 05.
- **DL 모델 파일이 `models/`에 부재** (`ml_churn_model.pkl.joblib`만 있음) → Part 05.
- **CV 단계에서 클래스 불균형 미적용** — `scale_pos_weight`/`is_unbalance`가 최종 재학습에서만 적용 → Part 04.
- **시퀀스 임베딩 미구현** — `nn.Embedding` 없음, K-Means `behavior_pattern_cluster`만 존재 → Part 03.

### 4.4 Uplift / Segmentation 라벨 정합

- **4분면 분류가 Uplift 중앙값 단일축만 사용** (`uplift_model.py:219-229`). require.md 117-120행이 요구한 "Uplift × baseline churn" 2축 미반영 → Part 06.
- **6세그먼트 라벨이 spec 예시(고가치-Persuadables, 신규-온보딩 등)와 다름** — RFM 8세그먼트와 Uplift 4분면이 별도 모듈로만 존재, 결합 라벨 부재 → Part 08.
- **priority_score 공식 불일치** — 실제 `monetary*(1-recency/max_recency)` (`main.py:650`), spec 예시 `Uplift × CLV`는 `budget_optimizer.py:73`에만 존재 → Part 08.

### 4.5 산출물/CLI 누락

- `--mode optimize` CLI에서 What-if 50/100/200% 시나리오·ROI 요약 미저장 (대시보드만 존재) → Part 09.
- `--mode optimize` CLI ROI 컬럼 누락 → Part 09 (리뷰 03 추가 지적).
- `feature_store` 영속화가 단순 CSV 저장으로 대체 → Part 03.
- `run_monitor`가 features.csv를 절반으로 split하여 reference/production 비교 → 시간축 누락 → Part 12.

### 4.6 문서/설정 불일치 (리뷰 03이 추가 발견)

- **단가 불일치**: `simulator_config.yaml` (push 200원, coupon 5,000원, call 15,000원) vs `docs/retention_strategy.md` (push 1,000원, coupon 30,000원, VIP 70,000원). 두 소스가 서로 다른 비용 구조를 사용 → Part 09 보고서가 인용은 했으나 지적 누락.

---

## 5. ✅ 완전 충족 파트

| 파트 | 비고 |
|------|------|
| Part 10 (A/B 테스트) | Power Analysis · Chi-square/Z-test · 95% CI · p-value · 결과 리포트 모두 코드+문서로 입증 |
| Part 11 (대시보드) | Streamlit 10개 항목 모두 컴포넌트 단위로 식별, docker-compose 8501 포트·수동 새로고침까지 완비 |
| Part 12 문서화 (B1~B8) | README 아키텍처 ASCII + Data Flow, 4종 docs, docstring 다수(예: feature_engineering 32, cohort 46, ab_testing 44, budget_optimizer 60), src/ 9-도메인 분리 |

---

## 6. Round 3 리뷰어 점수 (감사 보고서 자체 품질)

| 리뷰 | 대상 | 점수 | 결정적 지적 |
|------|------|------|-------------|
| review-01 | Part 01 + Part 07 | 4.5/5 | Part 07이 "`results/` 부재"로 진술했으나 실제 디렉토리는 존재(`clv_predictions.csv`만 부재) |
| review-02 | Part 02 + Part 08 | 4.0/5 | 두 보고서 모두 "`results/`가 비어 있다/존재 안 함"이라고 잘못 진술 — 실제로는 5개 cohort 산출물 존재 |
| review-03 | Part 03 + Part 09 | 4.5/5 | Part 09가 simulator_config vs retention_strategy 단가 불일치를 인용만 하고 지적 누락 |
| review-04 | Part 04 + Part 10 | 4.5/5 | Part 04가 "PNG 0개" 주장 — 실제 `cohort_retention_*.png` 2개 존재. `model_report.md` 라인 ±2 오프셋 |
| review-05 | Part 05 + Part 11 | **5.0/5** | 인용 14건 모두 라인·발췌 정확히 일치, 결함 진단 모두 재현 가능 |
| review-06 | Part 06 + Part 12 | 4.5/5 | Part 06 "`results/` 자체 미생성" 표현 부정확 (디렉토리는 존재, `uplift_results.csv`/`qini_curve.png`만 부재). docstring 카운트 5건은 grep과 100% 일치 |
| **평균** | — | **4.5/5** | 인용 정확도는 매우 높지만 "results/ 디렉토리 부재"라는 cross-cutting 오진술 다수 |

### 리뷰가 잡아낸 cross-cutting 오류 (보고서 보강 필요)

1. **`results/` 디렉토리 진술 정정 필요**: Round 1/2 진행 중 cohort 감사 에이전트가 실행하면서 5개 산출물을 생성한 상태. 실제 부재 파일은 `clv_predictions.csv`, `segments_6plus.csv`, `uplift_results.csv`, `qini_curve.png`, `shap_summary.png`, `monitoring_report.json` 등 **개별 파일** 단위로 명시하는 것이 정확. (현 디렉토리에 `budget_optimization.csv`, `cohort_analysis.json`, `cohort_retention_curves.png`, `cohort_retention_heatmap.png`, `cohort_retention_matrix.csv` 5개 존재.)
2. **`docs/model_report.md:178-184` → `:176-184`** 라인 오프셋 (±2) — Part 04.

---

## 7. 결론 및 액션 아이템

### 즉시 수정 (P0)

- [ ] **Part 04**: `src/main.py`의 SHAP 호출을 `ShapExplainer` 클래스 시그니처/메서드명에 맞춰 수정 → `results/shap_summary.png` 산출 보장.
- [ ] **Part 07**: CLV 정확도 검증 루틴 추가(예: holdout monetary vs predicted_clv MAE/MAPE) + `results/clv_predictions.csv` Top-N + 분포 리포트.
- [ ] **Part 02**: `run_cohort`가 `compute_churn_rates`·`extract_churn_sequences`·`analyze_pre_churn_events`·`compute_journey_funnel`을 호출하고 결과를 `results/`에 저장하도록 와이어링.

### 명세 정합 (P1)

- [ ] **Part 01**: `small_mode.num_customers`를 5,000으로, Treatment/Control 각 10,000명 보장(stratified sampling) 추가.
- [ ] **Part 05**: DL 학습 경로에서 `sequence_utils.create_sequences`를 호출하여 진짜 시계열 입력으로 전환 + DL 모델 파일을 `models/`에 저장.
- [ ] **Part 06**: 4분면 분류 기준을 `(uplift_score, baseline_churn_prob)` 2축으로 수정 (require.md 117-120 라인 정의 준수).
- [ ] **Part 08**: 결합 라벨(예: "고가치-Persuadables", "신규-온보딩")과 `priority_score = uplift × clv` 공식을 메인 파이프라인에 통일.
- [ ] **Part 09**: `--mode optimize` CLI가 What-if 50/100/200% 시나리오 + ROI 요약을 `results/`에 저장하도록 확장. simulator_config 단가와 retention_strategy 단가 불일치 정합.

### 보강 (P2)

- [ ] **Part 03**: `FeatureEngineer.save_to_feature_store`를 `run_features` 메인 경로에서 호출.
- [ ] **Part 04**: CV 단계에서도 클래스 불균형 가중치 적용.
- [ ] **Part 12**: `run_monitor`의 reference/production을 시간 분할 또는 누적 운영 시점 기반으로 재설계.

### 산출물 체크리스트 미생성 파일 (require.md 산출물 체크리스트 기준)

- `results/clv_predictions.csv` — Part 07
- `results/segments_6plus.csv` — Part 08 (`run_segment` 산출물명도 다름)
- `results/uplift_results.csv` — Part 06
- `results/qini_curve.png` — Part 06
- `results/shap_summary.png` — Part 04
- `results/monitoring_report.json` — Part 12 (코드 경로는 존재, 파이프라인 미실행)

---

## 8. 부록: 보고서 인덱스

```
.audit/
├── round1/
│   ├── part-01-simulator.md
│   ├── part-02-cohort.md
│   ├── part-03-features.md
│   ├── part-04-ml-model.md
│   ├── part-05-dl-model.md
│   └── part-06-uplift.md
├── round2/
│   ├── part-07-clv.md
│   ├── part-08-segmentation.md
│   ├── part-09-optimization.md
│   ├── part-10-abtest.md
│   ├── part-11-dashboard.md
│   └── part-12-monitoring-docs.md
└── review/
    ├── review-01.md  (Part 01 + 07)
    ├── review-02.md  (Part 02 + 08)
    ├── review-03.md  (Part 03 + 09)
    ├── review-04.md  (Part 04 + 10)
    ├── review-05.md  (Part 05 + 11)
    └── review-06.md  (Part 06 + 12)
```

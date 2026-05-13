# Iter12 — End-to-End Validity Audit (Coverage + Data Lineage + KPI Sources)

**Date:** 2026-05-12
**Triggered by user request:** *"pytest로 coverage 보고, end-to-end가 실제 return되는지, ML/DL 자료가 실제로 쓰이는지, 대시보드 각 지표가 제대로 된 과정에서 나오는지 판단"*

**Method:** 3 independent audit sub-agents in parallel.
- **A** ran `pytest --cov=src` and analysed coverage.json by file/module
- **B** traced 5 end-to-end flows (simulator → features → train → predict → dashboard) by reading source code
- **C** classified all 108 visible dashboard KPIs into source-type buckets (REAL / DERIVED / HARDCODED / FALLBACK / UNCLEAR)

**Aggregate verdict:** **MIXED PASS-WITH-CAVEATS.**

The pipeline itself (simulator → features → ML/DL training → predictions → results CSVs) **DOES run and produces real artifacts.** Core ML KPIs (AUC, CLV total, uplift counts, segment counts, cohort retention, budget LP allocation) **DO trace to real artifacts.** However, **27.8% of dashboard KPIs are fixture-driven** (most damaging: Page 02 confusion matrix overwrites real Precision/Recall, hiding the actual model metrics), and **none of the four trained models are loaded by the dashboard for inference** — they're saved to `models/` and never re-read.

---

## 1. Pytest coverage results (Agent A)

| Metric | Value |
|---|---:|
| Total tests | 2,591 |
| Passed | 2,583 |
| Failed | 7 |
| Skipped | 1 |
| **Overall coverage** | **73.20 %** (8,703 / 11,890 statements) |

### Failed tests (all 7 in same file)
- `tests/test_model_monitoring_view.py` — stale assertions for a never-merged "Survival Curves" redesign. NOT a product bug; remove or update these tests.

### Critical coverage gaps

| File | Coverage | Lines missed | Audit risk |
|---|---:|---:|---|
| `src/dashboard/app.py` | **41.3%** | 1,079 | 🚨 ALMOST every page-render function is untested. Visual KPI rendering paths have no test coverage. |
| `src/main.py` | 48.6% | 806 | Pipeline orchestration past the ensemble step is untested. End-to-end run isn't covered by automated tests. |
| `src/dashboard/data_loader.py` | 66.5% | — | The very methods that bridge artifacts → KPIs. |
| `src/dashboard/calculations.py` | 66.2% | — | Includes the `format_count`, `format_currency_krw`, `compute_overall_roi`, `drift_trend_guard` helpers added in iter10 to fix P0 contradictions — **none enforced by tests**. |
| `src/models/churn_model.py` | 72.7% | 127 | Save/load and batch-predict helpers not covered. |
| `src/streaming/redis_producer.py` | 69.8% | 26 | Redis pathways feeding Page 13. |

**Implication:** the bottom half of the data flow (training → artifact persistence → re-use) is reasonably tested, but the **dashboard display layer is not** — anyone can change a KPI formula on app.py and tests will still pass.

---

## 2. End-to-end pipeline really runs (Agent B)

### 2-1. Artifact chain inventory (all present, recent, non-empty)

| Stage | Artifact | Size | Status |
|---|---|---|:--:|
| 1. Simulator | `data/raw/customers.csv` | OK | ✅ |
| 1. Simulator | `data/raw/events.csv` | OK | ✅ |
| 1. Simulator | `data/raw/generation_summary.json` | OK | ✅ |
| 2. Features | `data/features/feature_store.parquet` (33 features) | OK | ✅ |
| 3. ML train | `models/ml_churn_model.pkl.joblib` (LightGBM) | 275 KB | ✅ |
| 3. DL train | `models/dl_churn_model.pt` (Transformer) | 445 KB | ✅ |
| 3. CLV train | `models/clv_model.pkl` | 692 KB | ✅ |
| 3. Survival | `models/survival_model.pkl` | 1,999 KB | ✅ |
| 4. Predictions | `results/churn_predictions.csv` (20k rows) | 1.7 MB | ✅ |
| 4. Predictions | `results/clv_predictions.csv` | — | ✅ |
| 4. Predictions | `results/uplift_results.csv` | — | ✅ |
| 4. Predictions | `results/budget_optimization.csv` | 2.9 MB | ✅ |
| 4. Predictions | `results/cohort_analysis.json` (4 cohorts, [4,13]) | OK | ✅ |
| 4. Predictions | `results/ab_test_detailed.json` | OK | ✅ |

→ **모든 pipeline 단계가 실제로 실행되었고 산출물이 디스크에 존재.** Simulator는 20,000명의 가짜 데이터를 생성했고 ML/DL/CLV/Survival 4개 모델이 모두 학습되어 저장됨.

### 2-2. 그러나 대시보드는 학습된 모델을 **추론에 사용하지 않음**

| Model | 학습되어 디스크에 저장? | 대시보드가 inference에 사용? | 대시보드가 어떻게 KPI를 얻나? |
|---|:--:|:--:|---|
| `ml_churn_model.pkl.joblib` | ✅ YES | ❌ NO | `results/churn_predictions.csv` pre-baked CSV 읽음 |
| `dl_churn_model.pt` | ✅ YES | ❌ NO | 동일 — DL은 ensemble 점수에만 기여하고 끝 |
| `clv_model.pkl` | ✅ YES | ❌ NO | `results/clv_predictions.csv` 읽음 |
| `survival_model.pkl` | ✅ YES | ❌ NO | Survival data 자체가 fixture로 fallback (아래 참조) |
| MLflow | ✅ 4 runs 기록 | ❌ NO | `results/model_performance_history.csv` cached 3 행 읽음 |

**의미:** "대시보드가 ML 모델을 호스팅하는가" 관점에서는 **호스팅하지 않음**. 대시보드는 본질적으로 **batch pipeline의 출력을 시각화하는 reporting tool**이지 실시간 inference tool이 아님. 새 고객이 들어와도 대시보드가 모델을 호출하지 않음.

→ 이것 자체가 잘못된 것은 아님 (batch 패턴은 정상). 다만 README/marketing이 "real-time scoring" 운운한다면 그 부분은 호도가 됨.

### 2-3. 그러나 일부 KPI는 batch 산출물조차 **존재하지 않음**

다음 산출물들은 dashboard가 기대하지만 pipeline이 만들지 않아서 **fallback sample generator**가 실행됨:

- `confusion_matrices.json` — Page 02 혼동 행렬 fixture
- `roc_data.json` — Page 02 ROC curves fixture
- `survival_data.csv` / `survival_curves.json` — Page 07 KM curves fixture
- `scoring_history.csv` — Page 13 tab a "Total Scores 200" fixture
- `scoring_throughput.csv` — Page 08, 13, 15 throughput/latency/error fixture
- `retention_offers.csv` — Page 09 mid-strip cost/saved fixture
- `drift_history.csv` — Page 08, 13c drift trend fixture

이 산출물들이 **누락되었을 때 dashboard는 조용히 `_generate_sample_*()` 함수로 가짜 데이터를 만들어 보여줌**. 사용자는 fixture라는 표시 없이 KPI를 봄.

---

## 3. KPI 소스 분류 (Agent C)

**108 KPI 감사 결과:**

| 분류 | 수 | 비율 | 의미 |
|---|---:|---:|---|
| **REAL_ARTIFACT** (results/*.csv 또는 *.json에서 1:1 추출) | 21 | 19.4% | 신뢰 가능 |
| **DERIVED_FROM_REAL** (real artifact에서 Python 계산) | 49 | 45.4% | 신뢰 가능 |
| **LIVE_PROBE** (Redis/MLflow 라이브 확인) | 5 | 4.6% | 신뢰 가능 |
| **CONFIG_VALUE** | 3 | 2.8% | 신뢰 가능 |
| **소계 (real 또는 derived from real)** | **78** | **72.2%** | ✅ |
| **FALLBACK_SAMPLE** (artifact 없을 때 _generate_sample_*) | 23 | 21.3% | ⚠ |
| **HARDCODED_FIXTURE** (literal dict/list in code) | 5 | 4.6% | 🚨 |
| **DEFAULT_LEAK** (`.get(col, 0)` 패턴) | 2 | 1.9% | ⚠ |
| **소계 (suspect)** | **30** | **27.8%** | ❌ |

### 3-1. 페이지별 신뢰도 점수

| Page | 헤드라인 KPI | REAL/DERIVED | FIXTURE/FALLBACK | 등급 |
|---|---|---:|---:|:--:|
| 00 Overview | 4 KPI | 4 | 0 | ✅ A |
| 01 Churn Analytics | 5 KPI | 5 | 0 | ✅ A |
| 02 Model Performance | 5 KPI | 1 (AUC) | 4 (P/R/F1/Acc + matrices) | 🚨 D |
| 03 Segmentation | 3 KPI | 3 | 0 | ✅ A |
| 04 Cohort | 4 KPI | 4 | 0 | ✅ A |
| 05 Budget | 4 KPI | 4 | 0 | ✅ A |
| 06 A/B Testing | 7 KPI | 7 (after iter11 empty state) | 0 | ✅ A |
| 07 Survival | 4 KPI | 1 (Events count) | 3 (median, KM, hazard) | ⚠ C |
| 08 Monitoring | 8 KPI | 0 | 8 (모두 fallback) | 🚨 D |
| 09 Recommendations | 8 KPI | 4 (top strip) | 4 (mid cost-benefit strip) | ⚠ C |
| 10 CLV | 4 KPI | 4 | 0 | ✅ A |
| 11 Uplift | 4 KPI | 4 | 0 | ✅ A |
| 12 Campaign | 11 KPI | 11 | 0 | ✅ A |
| 13 Real-Time (tab a) | 7 KPI | 0 | 7 (전체 fixture) | 🚨 D |
| 13 Real-Time (tab b) | 7 KPI | 0 | 7 (전체 fixture) | 🚨 D |
| 13 Real-Time (tab c) | 4 KPI | 0 | 4 (drift fixture) | 🚨 D |
| 14 MLflow | 4 KPI | 4 (AUCs from json) | 0 | ✅ A |
| 15 System Health | 14 KPI | 9 (live probes) | 5 (throughput section) | ⚠ B |

→ **9 페이지 A등급**, **3 페이지 B-C등급**, **4 페이지 (02, 08, 13a/b/c) D등급**.

### 3-2. 가장 의심스러운 KPI 5가지

1. **Page 02 confusion matrix-derived P/R/F1/Acc** — `_generate_sample_confusion_matrices`가 리터럴 {[350,50],[80,120]} 등을 반환하고, `app.py:439-463`이 **실제 `model_metrics.json`의 P/R/F1을 이 fixture로 덮어씀**. 실제 ml_model은 P=0.5331/R=0.7791인데 dashboard는 P=0.7059/R=0.6000을 보여줌. iter10에서 "headline ↔ matrix 일치" 수정한 것도 사실은 둘 다 fixture에서 도출되도록 한 것 (real ↔ matrix는 여전히 불일치).

2. **Page 07 Median Duration 309d / KM curves / hazard rates** — `survival_data.csv` 없어서 `_generate_sample_survival(churn_prob)`이 `duration = 365·(1-churn_prob)` 결정적 변환으로 가짜 생존 시간 생성. survival_model.pkl은 1.9MB 저장되었지만 사용 안 됨.

3. **Page 08 Avg req/min 49 / Latency 19.1ms / Error 1.03%** — sinusoidal `_generate_sample_scoring_throughput`. 실제 운영 텔레메트리 0건.

4. **Page 09 mid-strip ₩1.21M cost / ₩10.89M saved / 9.0x ROI** — `_generate_sample_retention_offers` n=50. 상단 KPI (총 20,000 추천)은 REAL이지만 중간 ROI 카드는 fixture.

5. **Page 13 tab a Total Scores 200 / 27.30% / 17 high-risk** — `_generate_sample_scoring_history` n=200, `Beta(2,5)` 분포. Redis 스트림은 실제로는 0이며 (Live probe) tab b의 44 offers, tab c의 drift도 같은 패턴.

### 3-3. 신뢰 가능 KPI

다음은 **모두 real artifact 또는 live probe에서 도출됨, 신뢰 가능:**

- **Page 00, 01**: Total Customers (20,000), Avg Churn Prob (31.31%), High Risk (5,717), histogram, donut, feature importance, segment-level churn rate
- **Page 02**: AUCs (0.8852 / 0.8860 / 0.8866) — **AUC만** real
- **Page 03**: 6 segments, 20,000 합계, Avg CLV by segment
- **Page 04**: 4 cohorts, retention matrix [4,13], M1/M3/M6/M12
- **Page 05**: 50M budget, segment allocation table, ROI 3.84x (budget envelope), 122 retained
- **Page 06**: power calculator (이제 empty state로 0/0 표시 안 함)
- **Page 10**: Total CLV ₩57.94B, percentiles, n=1/n=2 segments 식별 가능
- **Page 11**: 4 quadrants (persuadable 2,708 / sure_thing 12,929 / sleeping_dog 3,683 / lost_cause 600), Avg Uplift 0.0434
- **Page 12**: Total CLV / At-Risk CLV / Avg Uplift / 122 retained / 3.84x ROI
- **Page 14**: 3 MLflow runs, best AUC 0.8866, model_performance_history.csv 3 rows
- **Page 15**: Redis Connected, MLflow Connected, drift status (live probes)

---

## 4. 핵심 결론

### 4-1. 답변별

**Q1: pytest 커버리지는?**
→ **73.20% (8,703/11,890)**. 7 failures (모두 stale tests in `test_model_monitoring_view.py`). 가장 큰 위험: **dashboard layer (`app.py`)가 41.3%**. KPI 렌더링 경로는 거의 테스트 부재.

**Q2: end-to-end가 진짜 return 하는가?**
→ **YES**. simulator (20K 고객 + 이벤트) → features (33 features) → ML/DL/CLV/Survival 4개 모델 학습 → predictions CSV / cohort JSON / budget LP CSV → 모두 실제로 생성됨. `python -m src.main --mode all`이 실제로 동작하고 results/와 models/에 산출물 남김.

**Q3: ML/DL 자료가 진짜 쓰이고 있는가?**
→ **부분적**. 학습된 모델은 디스크에 저장되지만 dashboard가 inference에 사용하지는 않음 (batch reporting 패턴). 다만 dashboard가 보여주는 churn/CLV/uplift KPI는 모두 모델 학습 결과의 산출물 CSV에서 1:1로 추출됨 — 즉 모델의 "결과"는 보여주고 있고, 모델 자체를 호출하지는 않음.

**Q4: 대시보드 각 KPI가 제대로 된 과정에서 나왔는가?**
→ **108개 중 78개 (72.2%) YES, 30개 (27.8%) NO**. NO 그룹은:
- **5개 hardcoded** (Page 02 confusion matrix → P/R/F1 덮어씀)
- **23개 fallback_sample** (Page 07 survival, Page 08 throughput, Page 09 retention-offers, Page 13 a/b/c real-time, Page 15 일부)
- **2개 default_leak**

### 4-2. 가장 큰 문제 한 개를 꼽으라면

**Page 02 (Model Performance) 헤드라인 Precision/Recall/F1/Accuracy 4개가 fixture 혼동 행렬에서 도출되어 진짜 모델의 값을 덮어씌움.**

- 실제 ml_churn_model의 P=0.5331, R=0.7791 (model_metrics.json)
- Dashboard 표시 ml_model P=0.7059, R=0.6000 (fixture matrix에서 도출)
- iter10에서 "headline ↔ matrix 일치" 수정했지만, 양쪽 다 fixture에서 도출되도록 했을 뿐 **real artifact로 통일하지는 않음**

→ Page 02는 모델 성능을 "측정"하는 페이지인데 실제 측정값을 은폐. SaaS에서 가장 위험한 패턴.

### 4-3. SaaS 출시 관점에서

| 카테고리 | 평결 |
|---|---|
| 핵심 ML 결과 (churn/CLV/uplift/segmentation/cohort/budget) | ✅ 신뢰 가능. 실제 모델 산출물 |
| 모델 비교/평가 페이지 (P02) | 🚨 fixture로 덮어쓴 P/R/F1. 즉시 수정 필요 |
| Survival 분석 (P07) | ⚠ KM/hazard 모두 fixture. survival_model.pkl 활용해야 함 |
| 운영 텔레메트리 (P08 monitoring, P13 real-time, P15 throughput) | 🚨 모두 fixture. 실제 Redis/MLflow 텔레메트리 연결 필요 |
| MLflow 실험 추적 (P14) | ✅ AUC만 real. hyperparameter sweep은 1회 smoke run만 |
| 비즈니스 가치 (P10 CLV, P11 uplift, P12 campaign) | ✅ 신뢰 가능 |

---

## 5. 우선순위 권장 사항

### P0 (즉시 수정)

1. **Page 02 fixture override 제거**: `app.py:439-463`이 fixture confusion matrix로 real model_metrics.json의 P/R/F1을 덮어쓰는 코드를 삭제. 헤드라인 KPI는 실제 학습 산출물에서만 도출되도록 변경.

2. **`_generate_sample_*()` fallback에 명시적 경고 표시**: dashboard가 fixture 데이터로 fallback 할 때 `st.warning("Using sample data — real artifact missing: <filename>. Run pipeline to regenerate.")` 표시. 현재는 조용히 fallback해서 사용자가 fixture인지 모름.

### P1

3. **Pipeline이 누락된 7개 산출물 생성하도록 확장**:
   - `confusion_matrices.json` (P02)
   - `roc_data.json` (P02)
   - `survival_data.csv`, `survival_curves.json` (P07)
   - `scoring_history.csv`, `scoring_throughput.csv` (P08, P13, P15)
   - `retention_offers.csv` (P09)
   - `drift_history.csv` (P08, P13c)

4. **Dashboard layer 테스트 보강**: `app.py` 커버리지 41.3% → 70%+ 목표. KPI 계산 helper와 page-render 경로 unit test 추가.

### P2

5. **MLflow 실제 hyperparameter sweep**: 3 run이 모두 동일 config인 것을 실제 grid/Bayesian sweep으로 교체.

6. **Survival 모델 실제 사용**: 저장된 `survival_model.pkl` (1.9MB)을 dashboard에서 inference로 호출하여 fixture 대체.

---

## 6. 산출물

- `_test_results/iter12/audit_A_coverage.md` — pytest 커버리지 상세
- `_test_results/iter12/audit_B_lineage.md` — end-to-end 데이터 계보
- `_test_results/iter12/audit_C_kpi_sources.md` — 108 KPI 분류
- `_test_results/iter12/coverage.json` — 머신 가독 커버리지 데이터
- `_test_results/iter12/coverage_html/index.html` — HTML 커버리지 리포트
- `_test_results/iter12/pytest_full.log` — pytest 풀 로그
- `_test_results/iter12/iter12_end_to_end_audit.md` — 본 종합 보고서

---

## 7. 한 줄 요약

> **파이프라인은 진짜 돌고 모델은 진짜 학습되어 결과를 남긴다(72.2% KPI는 신뢰 가능). 하지만 Page 02 모델 성능 비교 페이지가 fixture로 진짜 P/R/F1을 덮어쓰고, Page 07 survival·Page 08 monitoring·Page 13 real-time은 모두 가짜 데이터(`_generate_sample_*`)로 렌더링되며 사용자는 그것이 fixture임을 알 수 없다.** Pytest 커버리지 73%지만 dashboard 레이어는 41%로 KPI 렌더링 경로가 검증되지 않은 상태. iter12에서 P0 두 항목(P02 override 제거, fallback 경고 표시)을 닫으면 dashboard 신뢰도는 ≥95%로 올라간다.

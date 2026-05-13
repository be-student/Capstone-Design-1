# iter16 — 사용자 6개 의심사항 조사 결과 + 수정 계획

**Date**: 2026-05-13
**Method**: 6개 read-only sub-agent 병렬 조사 → 메인 세션이 코드 수정
**Scope**: Dashboard P00 (Overview) / P01 (Churn Analytics) / P02 (Model Performance) / P04 (Cohort) / P07 (Survival) / P09 (Recommendations) / P15 (System Health)

---

## 사용자 원 질문

1. 개요에 왜 개별고객 조회의 추천 액션이 다 N/A야?
2. 평균 이탈 확률은 어떤걸 의미하는 거야(31.31%) — PRD에서 15~25%를 유지하는 고객 시뮬레이터의 평균 이탈 확률을 대시보드에 올려야 하는 거 아니야?
3. 모델 성능 ml, dl, 앙상블이 AUC가 너무 비슷한게 의심돼
4. 코호트 분석 대시보드는 뭘 의미하는 거야?
5. 생존 분석은 뭘 의미하고 그 중에서 업리프트 세그먼트별 평균 생존 확률은 왜 다 비슷한 값을 가져?
6. 추천에는 ValueError가 나 있어 / 시스템 헬스를 보니까 문제가 감지되었대 확인해봐

---

## ① Overview 개별고객 추천액션이 모두 "N/A"

### 원인
`src/dashboard/app.py:350` — Customer Lookup이 `data_loader.load_predictions()` 결과(`churn_predictions.csv`)만 사용.

`churn_predictions.csv` 컬럼:
```
['customer_id', 'churn_probability', 'risk_level', 'persona', 'segment', 'split', 'prediction_source']
```
→ **`recommended_action` 컬럼이 존재하지 않음**.

`src/dashboard/app.py:506`:
```python
action = row.get("recommended_action", "N/A")  # 항상 fallback
```

실제 추천 데이터는 `results/retention_offers.csv`에 존재 (`no_action: 16,602 / coupon: 3,398`). 두 CSV의 `customer_id` 교집합은 **100% (20,000명)**.

### 영향
전 고객 20,000명 (100%)이 N/A 표시.

같은 패턴으로 `clv_predicted`(line 505) / `days_since_last_purchase`(line 507)도 fallback 값(0 / 0d) 표시 중.

### 수정 방향
`DashboardDataLoader.load_predictions()`에서 retention_offers + features의 일부 컬럼을 사전 머지하여 통합 view 반환. customer_id left join, 미존재 시 `recommended_action="no_action"` fillna.

---

## ② 평균 이탈 확률 31.31% vs PRD 15-25%

### 31.31%의 의미
`src/dashboard/app.py:366`:
```python
avg_churn = predictions["churn_probability"].mean()
```
= 모델(앙상블 ML 0.6 + DL 0.4)이 출력한 **예측 확률의 산술 평균** (`results/churn_predictions.csv`의 `churn_probability` 컬럼).

| 통계량 | 값 |
|---|---:|
| Mean | **0.3131** |
| Median | 0.1539 |
| Std | 0.3236 |
| 25% / 75% | 0.0358 / 0.5706 |
| Label=0 평균 | 0.201 |
| Label=1 평균 | 0.764 |

모델이 우편향(right-skew) 분포를 학습 → mean이 base rate보다 위로 끌려감.

### PRD 15-25%의 의미
**라벨 이탈률(ground-truth)**, 즉 시뮬레이터가 생성한 데이터의 `churn_label` 평균.

근거:
- `config/simulator_config.yaml:47-49` — `target_churn_rate: {min: 0.15, max: 0.25}`
- `src/data/orchestrator.py:215` — `customers_df["churn_label"].mean()` 으로 검증
- **실측: `results/features.csv` `churn_label.mean() = 19.99%`** → PRD 충족 ✅

### 평결
**버그 아님 / UX·네이밍 결함**.

라벨 이탈률(19.99%)이 PRD를 정상 만족하는데, 대시보드가 이 값을 어디에도 노출하지 않고 예측 확률 평균(31.31%)만 "Avg Churn Prob"라는 모호한 라벨로 보여줘 사용자가 PRD 위반으로 오해.

### 사용자 지시 (이번 수정에 반영)
> 개요와 이탈분석에서 나타내는 평균 이탈 확률은 생성한 고객 시뮬레이터의 확률로 수정

**수정 방향**: Overview + Churn Analytics 페이지의 "Avg Churn Prob" KPI의 값을 `features["churn_label"].mean()` (= 시뮬레이터가 만든 실제 이탈률, 19.99%)로 교체. 라벨도 적절히 수정.

---

## ③ ML / DL / Ensemble AUC 너무 비슷

| 모델 | AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ML (LightGBM/XGBoost) | **0.8852** | 0.5331 | 0.7791 | 0.6331 |
| DL (Transformer, best epoch 9) | **0.8860** | 0.6759 | 0.6318 | 0.6531 |
| Ensemble (0.6·ML + 0.4·DL) | **0.8866** | 0.6426 | 0.6621 | 0.6522 |

AUC 셋째 자리(0.88X)까지 동일, 차이는 0.00136 (0.15%).

### 평결: 버그 아님 / 의도된 동작 + 진단된 한계

**라벨 누수 차단 확인**:
- `src/main.py:1359-1370` `_observation_cutoff()` + `_filter_events_to_cutoff()` 명시적 적용
- DL이 0.99 같은 의심값이 아닌 0.886에 머무름 → 누수 부재 뒷받침

**진짜 원인**:
- **Feature 공유**: `_feature_cols(X_train)` (main.py:1507) — DL과 ML이 33개 동일 static feature를 공유. DL은 그 위에 monthly panel 6개만 추가. epoch 0에서 이미 val_auc 0.882로 시작 → DL의 시퀀스 정보 기여 미미.
- **데이터 천장**: churn 시그널이 recency/frequency 같은 정적 feature에 강하게 응축 → 모델 종류 무관하게 비슷한 ceiling.
- **Ensemble의 한계**: 상관 높은 두 점수의 가중평균은 한계 개선 미미 (수학적으로 정상).

### 수정 방향
모델 아키텍처는 그대로 두되, **Page 02에 진단 disclaimer + ML/DL 예측 확률 Pearson 상관**을 표시하여 사용자가 유사 AUC가 "버그가 아닌 데이터 천장 + feature 공유"임을 이해할 수 있게 함.

---

## ④ 코호트 분석 페이지 의미

**한 줄 정의**: 같은 시기 가입한 고객 집단(cohort)이 시간이 흐르면서 얼마나 남아 있는가를 추적.

### 주요 차트 / KPI (7개)

1. **KPI 4종**: Total Cohorts / Periods Tracked / Avg Period-1 Retention / Avg Final Retention
2. **Retention Heatmap** (Cohort × Period) — 색상으로 retention 한눈에 비교
3. **Retention Curves by Cohort** — 곡선 위쪽 = "좋은 cohort"
4. **Average Retention Curve** — 전체 평균 곡선 (LTV·CAC 페이백 추정용)
5. **Cohort Sizes (Period-0)** — 월별 신규 고객 수 (마케팅 인풋 규모)
6. **Period-over-Period Retention Change** — 큰 음수 막대 = "끊기는 hook 구간"
7. **Raw Retention Matrix** — 감사·다운로드용

### 비즈니스 사용
- **마케팅 채널 ROI 검증**: 새 cohort 곡선이 기존보다 낮으면 CAC 회수 불가 → 채널 예산 재배분
- **제품 변경 효과 측정**: 기능 출시 후 합류 cohort retention 위로 올라오면 효과 입증
- **이탈 hook 식별**: PoP drop 큰 구간에 갱신·결제·온보딩 이벤트 점검
- **LTV 추정**: Avg retention curve 적분으로 LTV 추정 → CAC 상한선 결정

### 현재 상태
✅ Real artifact 사용 중 (`cohort_retention_matrix.csv` 4×13)
- 1개월 잔존 ~98.9%, 6개월 ~92-94%, 마지막 ~86-89% — SaaS 평균 수준
- ⚠ "Limited cohort window" 경고 (4개뿐, 6-12 권장)
- ⚠ 4월 cohort 음수 churn `-0.0114`

### 수정 사항: 없음 (정상 작동 중, 설명만 요청)

---

## ⑤ 생존 분석 + 업리프트 세그먼트별 평균 생존 확률 유사

### Survival Analysis 개념
이탈 사건까지 걸리는 시간을 모델링:
- **Kaplan-Meier (KM)**: 비모수 추정, 그룹별 경험 곡선
- **Cox PH**: 준모수 회귀, 공변량으로 개별 고객 S(t) 예측

### "Uplift Segment 평균 생존 확률" 막대가 모두 비슷한 원인

| 항목 | 값 |
|---|---|
| groupby 컬럼 (`app.py:2260`) | `survival.groupby("segment")` |
| **`segment` 실제 값** | bargain_hunter, dormant, explorer, new_customer, regular_loyal, vip_loyal — **behavioral 6 세그먼트** (uplift 아님!) |
| `survival_probability` 정의 | `sp_90.round(6)` = Cox PH의 **S(90일)** |
| Segment별 평균 spread | **0.93pp** (max-min, 모두 0.99 근처 ceiling) |
| **같은 데이터 S(365)의 spread** | **88.1pp** (dormant 4.55% ↔ vip_loyal 92.69%) |

### 원인 (3가지 복합)
1. **시점 t=90일이 너무 이름** — `duration_days` 최솟값 198일이라 90일 시점에선 거의 이탈 없음 → S(90)이 ceiling
2. **차트 라벨 버그** — 제목/캡션은 "Uplift Segment"라 표기하지만 실제 groupby는 behavioral 6-segment
3. **평균 집계가 분산 압축** — segment 내부 std 0.001~0.003 수준

### 평결: 모델은 정상 / 차트 표시 정책 결함

### 수정 방향
1. `app.py:2260-2274`에서 `survival_probability`(S90) 대신 **`survival_prob_365d`** 사용
2. 차트 제목을 **"Average Survival Probability by Behavioral Segment"** 로 정정

---

## ⑥-A 추천 페이지 ValueError

### 위치
`src/dashboard/recommendations_view.py:716` (`_render_cost_benefit_analysis`)

### 오류
```
ValueError: Invalid element(s) received for the 'size' property of scattergl.marker
```

### 원인
산점도 `px.scatter(..., size="expected_uplift", ...)` 호출인데, `retention_offers.csv`의 `expected_uplift` 컬럼이 **3,683행 (18.4%) 음수** (min = -0.778). Plotly marker size는 `[0, inf)`만 허용 → ValueError로 페이지 렌더링 중단.

음수 uplift는 `no_action` 행에서 발생하는 정상 데이터 (uplift 모델이 부정 효과 예측한 고객 = 처치 미시행).

### 영향
페이지 마지막 산점도에서 사망. 상단 KPI / donut / box plot / ROI-by-type 차트는 정상이지만 그 아래는 unreachable.

### 수정 방향
`size`로 사용할 컬럼을 `clip(lower=0)` 또는 별도 `size_metric` derived 컬럼 사용. 음수는 0으로 축소.

---

## ⑥-B 시스템 헬스 "문제 감지"

### Worst-child rollup
mlflow=DOWN ∨ drift=DOWN → **overall=DOWN → 적색 배너 "System Issues Detected"**

| Subsystem | 표시 상태 | 실제 상황 | 평결 |
|---|---|---|---|
| redis | HEALTHY | ping OK | ✅ 정상 |
| pipeline | HEALTHY | artifact 51개, model 4개 | ✅ 정상 |
| **mlflow** | **DOWN** | `mlflow.search_experiments()`가 **`ModuleNotFoundError: alembic`** 으로 실패. MLflow 2.12.1 import 자체는 성공, sqlite 파일 존재, 실제로는 정상. dashboard 컨테이너 이미지에 `alembic` 패키지 누락 + `except ImportError`가 모듈명 확인 없이 catch해서 UI에 "mlflow package not installed"라는 **오해성 메시지** | ❌ False alarm |
| **drift** | **DOWN** | `drift_history.csv` 마지막 행이 `feature_name=__overall__`, `alert_level=red`, `psi_mean=0.002139`. 모든 개별 feature는 green/yellow, PSI 평균 0.002로 yellow_threshold=0.10보다 한참 아래. **`is_initial_check=True` 행이라 baseline 부재로 conservative red 표기**, `get_system_health_summary`가 `iloc[-1]`만 보고 DOWN으로 매핑 | ❌ False alarm |

### 수정 방향
1. `Dockerfile.dashboard`에 `alembic` 패키지 추가
2. `check_mlflow_health`의 `except ImportError` 분기 좁히기 — 실제 모듈명 노출
3. Drift writer가 `__overall__` 행에 alert_level 산정 시 `is_initial_check=True`이거나 `psi_mean < red_threshold`이면 red 라벨 금지
4. `get_system_health_summary`의 drift 평가를 `iloc[-1]` 대신 `feature_name=='__overall__' & is_initial_check==False` 필터로 제한

---

## 수정 우선순위

| 이슈 | 영향 범위 | 우선순위 | iter16 처리 |
|---|---|:--:|:--:|
| ⑥-A P09 ValueError | 페이지 일부 렌더 중단 | 🔴 P0 | ✅ FIX |
| ⑥-B mlflow/drift false alarm | 헬스 배너 false positive | 🟡 P1 | ✅ FIX |
| ① Overview N/A | 전 고객 100% N/A 표시 | 🟡 P1 | ✅ FIX |
| ⑤ Uplift segment 차트 misleading | 라벨 잘못 + S(90) 의미 없는 시점 | 🟡 P1 | ✅ FIX |
| ② Avg Churn Prob → 시뮬레이터 라벨 | UX 오해 (사용자 명시 지시) | 🟠 P2 | ✅ FIX |
| ③ AUC 유사성 | 데이터 천장 (의도된 동작) | 🟢 P3 | ✅ DIAGNOSTIC ADD |
| ④ 코호트 설명 | 정상 작동 중 (설명만) | — | ⏭ SKIP |

---

## 수정 계획

병렬 sub-agent + 메인 세션 분담:

| Fix | 파일 | 담당 |
|---|---|---|
| #1 | `src/dashboard/data_loader.py` (load_predictions) | Agent A |
| #2 | `src/dashboard/app.py` + i18n KO | Main |
| #3 | `src/dashboard/app.py` (Page 02) | Main |
| #5 | `src/dashboard/app.py` (Page 07) | Main |
| #6-A | `src/dashboard/recommendations_view.py` | Agent C |
| #6-B | `src/dashboard/system_health_view.py` + `src/monitoring/monitoring_service.py` + `Dockerfile.dashboard` | Agent D |

app.py 변경(#2, #3, #5)은 메인 세션이 순차 처리하여 conflict 방지.

# A2 — 세그멘테이션(Segmentation) / 코호트(Cohort) / 예산 최적화(Budget)

페이지 03, 04, 05에 대한 독립 감사. 이미지 + 구조화 데이터 덤프를 페이지 내부 산술과 교차 검증함.

---

## Page 03 — Customer Segmentation

### 화면에 보이는 KPI
- Total Segments: **6**
- Total Customers: **20,000**
- Highest Risk Segment: **dormant**
- 세그먼트 비율 (도넛): regular_loyal 24.7%, bargain_hunter 20.4%, new_customer 15.1%, explorer 14.9%, dormant 14.7%, vip_loyal 10.2%
- 세그먼트 카운트 (막대): vip_loyal 2,030 | regular_loyal 4,949 | bargain_hunter 4,087 | explorer 2,975 | new_customer 3,014 | dormant 2,945
- Mean CLV (KRW): vip_loyal 12,760,815 | regular_loyal 3,248,249 | bargain_hunter 1,932,498 | new_customer 1,503,177 | explorer 1,120,117 | dormant 66,362
- Definitions 표 행: vip_loyal, loyal_customer, potential_loyalist, at_risk, hibernating, explorer, new_customer, bargain_hunter (8행)

### 잘못된 / 신뢰성 부족
- **Definitions 표 불일치 (네이밍 분열).** Segment Definitions & Retention Actions 표에는 `vip_loyal, loyal_customer, potential_loyalist, at_risk, hibernating, explorer, new_customer, bargain_hunter` (8개 이름) 명시. 하지만 Distribution / Statistics 차트는 `vip_loyal, regular_loyal, bargain_hunter, new_customer, explorer, dormant` (6개 이름) 사용. **중복되는 이름은 4개뿐** (vip_loyal, explorer, new_customer, bargain_hunter). `regular_loyal`과 `dormant`는 Definitions 표 어디에도 없고, 반대로 `loyal_customer / potential_loyalist / at_risk / hibernating`은 어떤 차트에도 나오지 않음. 헤드라인 "Highest Risk Segment = dormant"를 클릭한 사용자는 dormant를 Definitions 표에서 찾을 수 없음.
- **카운트 vs 비율 × 20,000** 모두 정합: 2,030+4,949+4,087+2,975+3,014+2,945 = **정확히 20,000** ✓; 비율% × 20,000도 보고된 카운트와 ±0.05pp 이내. 즉 샘플 크기 산술은 OK이지만, 차트의 6개 세그먼트를 설명하지 못하는 Definitions 표에 묶여 있음.
- **"Highest Risk Segment = dormant"** 옆에 위험 점수가 없음. 막대 차트를 읽어야 dormant가 ≈ 0.85+임을 알 수 있음. KPI가 정성적(qualitative)일 뿐임.
- **dormant의 Avg CLV = 66,362 KRW** (≈ $50). 다음 세그먼트(explorer 1.12M)보다 두 자릿수 낮음. dormant 코호트로서는 그럴듯하지만 주목할 만한 값 — 어떤 "가중" 롤업이든 이 극단 꼬리가 좌우함.

### 누락된 항목
- 헤드라인 KPI에 세그먼트 위험 점수가 없음 (라벨만 있음).
- CI / 샘플 가중 CLV 부재.
- 세그멘테이션 실행에 대한 날짜 / 모델 버전 스탬프 없음.
- Definitions 표가 한국어 컬럼 ("loyal_program", "engagement_campaign" …)으로 렌더링되는데 영문 fallback 없음; 부분적으로만 로컬라이즈됨.

---

## Page 04 — Cohort Analysis

### 화면에 보이는 KPI
- Total Cohorts: **4** (2024-01, 2024-02, 2024-03, 2024-04)
- Periods Tracked: **13**
- Avg Period-1 Retention: **99.0%**
- Avg Final Retention: **2.5%**

### 잘못된 / 신뢰성 부족
- **코호트가 4개뿐.** 어떤 종단(longitudinal) 내러티브에도 SaaS 표준인 ≥6–12 월별 코호트에 못 미침. 시뮬레이터의 n=20,000 고객 기준에서 이는 UI 한계가 아니라 데이터 생성 측 한계.
- **유지율 단조성(monotonicity) 위반 — Apr 2024 코호트: P7 = 91.0% → P8 = 92.1%.** 유지율 곡선은 정의상 비증가(non-increasing)여야 함 (같은 lookback 윈도우 내에서 이탈한 고객이 다시 살아날 수 없음). 이는 코호트 회계 버그이거나 페이지가 표시하지 않는 노이즈. 데이터 덤프에서 **CRITICAL 이상치**로 보고되었고 히트맵에서 시각적으로도 확인 가능.
- **"Avg Final Retention = 2.5%"는 쓰레기 평균.** 매트릭스가 우측 절단(right-truncated)되어 있음:
  - Jan 2024 P12 = 10.2%이지만 Feb의 P12, Mar의 P11–P12, Apr의 P10–P12는 모두 **0.0%** — 해당 코호트들이 **그만큼 오래 관측되지 않았기 때문**. 즉 NaN을 0으로 채운 것이지 실제 이탈이 아님.
  - 마지막 컬럼의 단순 평균 = (10.2 + 0 + 0 + 0) / 4 = **2.55%** → 표시된 2.5%와 일치. KPI가 0으로 채워진 미래 셀을 평균하고 있음을 확정.
  - 동일한 artifact가 Period-over-Period에도 오염: P9 −20.7%, P10 −23.4%, P11 −22.8%, P12 −21.5% — 모두 분모 붕괴(denominator-collapse) artifact, 실제 ~20pp 월별 이탈이 아님.
- **"마지막 살아남은" 셀들 자체가 비현실적**: Jan P12 = 10.2%, Feb P11 = 7.9%, Mar P10 = 9.8%, Apr P9 = 12.9% — 각 코호트가 우측 가장자리에서 한 기간에 ~80pp 떨어짐. 이는 데이터 윈도우 컷오프가 보이는 매트릭스로 누출된 것. 마스크 처리해야 하는데 그대로 표시됨.

### 누락된 항목
- 히트맵 / 곡선에 코호트별 샘플 크기 (n=?) 없음.
- 유지율 곡선에 CI 밴드 없음.
- 절단된 셀에 필터 / 마스크 없음; "as-of" 날짜가 없어 사용자가 Apr의 P10–P12가 미관측 셀임을 인지할 수 없음.
- 세그먼트 × 코호트 교차 표 없음 — Page 03의 `dormant` 세그먼트에서 코호트 뷰로 가는 경로가 없음.

---

## Page 05 — Budget Optimization

### 화면에 보이는 KPI
- Total Allocated: **50,000,000 KRW**
- Expected Retained: **118**
- Revenue Saved: **192,155,551 KRW**
- Avg ROI: **3.5x**

### 잘못된 / 신뢰성 부족
- **세그먼트 배분 합 ≠ Total Allocated (수정).**
  21,062,000 + 13,922,000 + 12,781,000 + 1,524,000 + 680,000 + 31,000 + 0 + 0 = **정확히 50,000,000 KRW** (데이터 덤프의 49M 메모는 자체 집계 오류였고, 재계산 결과 헤드라인 50M과 일치). 다만 **% 컬럼은 100.002%로 반올림** (42.124 + 27.844 + 25.562 + 3.048 + 1.36 + 0.062 + 0 + 0). 반올림 범위 내에서 세그먼트 배분은 정합. 합계는 **녹색**, 다만 데이터 덤프의 메모가 잘못이었음.
- **Avg ROI 3.5x ≠ 집계 ROI 3.84x.** Revenue Saved / Total Allocated = 192,155,551 / 50,000,000 = **3.843x**. 헤드라인은 **3.5x**라고 표시 — 즉 세그먼트별 ROI의 비가중 평균을 보여주고, 집계(매출/지출)는 3.84x. 동일 KPI strip에 두 가지 "ROI" 정의가 공존.
- **Baseline Retained 122 ≠ Allocation Summary Expected Retained 118.** "Expected Retained Customers by Scenario" 표에서 Baseline = 122, "Current Selection" = 122. 그런데 헤드라인 KPI는 **118**. 헤드라인은 Current Selection과 같은 시나리오(둘 다 50M, 배수 없음). 4명 차이 미설명.
- **high_value_persuadable**가 ROI 막대 ≈ **8x** (차트 최고)인데도 단지 **31,000 KRW (0.062%)**만 받음. LP에 표면화되지 않은 세그먼트 사이즈 캡이 있거나, 옵티마이저가 가중치를 잘못 주고 있음. 페이지에 "constraint binding" 인디케이터 없음.
- **Channel-Level Cost Breakdown** 섹션이 빈 H3로 렌더링되며 배너만: "Channel configuration not found in config. Add `budget.channels` to simulator_config.yaml…" — 빈 섹션 = 최종 사용자에게 보이는 구멍.
- **Cost Reduction 시나리오에서 Total Allocated = 50M (Baseline과 동일)이지만 별도 ROI 표시** — 막대 차트가 예산이 변하지 않았음에도 "Cost Reduction"을 별도 점으로 처리. 배수가 표시되지 않거나, 시나리오가 실제로 비용을 줄이지 않거나.

### 누락된 항목
- 유지 고객 점추정치에 CI 없음 ("Aggressive +50% retains 220"이 단일 숫자).
- LP 제약 또는 어떤 제약이 활성화되어 있는지 표시 없음 (세그먼트 캡, 최소 지출 등).
- 두 ROI 정의 간 조정(reconciliation) 부재; 헤드라인은 "세그먼트 ROI의 평균"이라는 표현 어디에도 없음.
- `budget.channels` 미설정 → 채널 분해 섹션 전체가 죽어 있음.

---

## 페이지 간 일관성 점검

| 점검 항목 | Page A | Page B | 결과 |
|---|---|---|---|
| Total customers 20,000 | 03 (KPI) | 04 (cohort base 암묵) | ✓ 시뮬레이터 n=20,000과 일치 |
| 세그먼트 이름 | 03 차트: 6개 | 03 Definitions 표: 8개 | ✗ **4개만 중복** |
| 세그먼트 이름 | 03: regular_loyal, dormant, bargain_hunter, … | 05: high_value_*, mid_value_*, low_value_*, sleeping_dog, lost_cause | ✗ **완전 분리된 taxonomy** — Page 03은 행동 세그먼트, Page 05는 uplift 세그먼트. 다리 없음. 사용자가 "`dormant`가 예산에서 어떻게 처리되고 있나?"를 물을 수 없음 |
| 헤드라인 "Expected Retained" | 05 KPI: 118 | 05 시나리오 표 Baseline / Current Selection: 122 | ✗ 같은 시나리오, 두 값 (Δ=4) |
| ROI 정의 | 05 KPI: 3.5x | 05 도출치: 192,155,551 / 50,000,000 = 3.843x | ✗ 정의 함정 |
| 코호트 커버리지 | 04: 4개 코호트뿐 (Jan–Apr 2024) | 03이 20,000 고객 존재 시사 | ✗ 시뮬레이터 데이터 양 대비 코호트 커버리지 빈약 |
| 유지율 단조성 | 04 Apr 2024: P7 91.0% → P8 92.1% | 보편적 코호트 수학: 비증가여야 함 | ✗ 위반, 표시 안 됨 |

---

## SaaS-readiness 평결

**평결: NOT SHIP-READY (출시 부적합).** 세 페이지에 걸쳐 독립적인 수치 무결성 결함 3건 (P03 세그먼트 이름 분열, P04 단조성 위반 + 헤드라인 KPI의 zero-truncation, P05 ROI/유지 카운트 불일치) + 세그멘테이션(P03)과 예산(P05) 간 완전한 taxonomy 단절. 임원용 SaaS 대시보드에서는 신뢰도 킬러 수준: 산술을 검증하는 분석가는 60초 안에 최소 한 가지 모순을 발견함.

**Top 3 블로커:**
1. **Page 04 — "Avg Final Retention 2.5%"가 0으로 채워진 미관측 미래 셀을 평균하고 있음** + 매트릭스에 유지율의 수학적으로 불가능한 단조성 위반 (Apr P7 91.0% → P8 92.1%). 절단 셀 마스킹, 단조성 버그 수정 또는 표기, KPI 재계산 필요.
2. **Page 05 — 한 카드 strip에 3개 KPI 내부 모순**: Avg ROI 3.5x vs 집계 3.84x; Expected Retained 118 vs Baseline 시나리오 122; high_value_persuadable이 ~8x ROI인데 지출의 0.062%만 배정 받음에 설명 없음. ROI 정의 하나로 통일, 유지 카운트 일치, LP 제약 노출 필요.
3. **Pages 03 ↔ 05가 분리된 세그먼트 taxonomy 사용** (behavioral vs uplift) + Page 03 자체의 Definitions 표가 차트 6개 세그먼트 중 4개만 커버. taxonomy를 통일하거나 매 페이지에 명시적 매핑과 함께 양쪽 모두를 표시. 현재 사용자는 `dormant`를 segmentation에서 budget까지 추적할 수 없음.

---

### 5줄 요약
1. Page 03 카운트는 정확히 20,000으로 합산되지만, Definitions 표가 8개 세그먼트 이름을 나열하는데 차트의 6개 세그먼트와 4개만 겹침 — `regular_loyal`과 헤드라인 "highest risk"인 `dormant`는 페이지 어디에도 정의되어 있지 않음.
2. Page 04는 코호트가 4개뿐, 유지율 단조성 위반 (Apr 2024: P7 91.0% → P8 92.1%) 포함, 헤드라인 "Avg Final Retention 2.5%"는 `(10.2+0+0+0)/4` — 0으로 채워진 미래 셀 평균.
3. Page 05의 "Avg ROI 3.5x"는 집계 192,155,551/50,000,000 = 3.843x와 모순, "Expected Retained 118"은 같은 페이지 Baseline/Current-Selection 시나리오 값 122와 모순.
4. Pages 03과 05는 완전히 분리된 세그먼트 taxonomy 사용 (behavioral vs uplift); 사용자가 단일 세그먼트를 두 페이지에 걸쳐 추적할 수 없음.
5. 평결: NOT ship-ready — 독립적 수치 무결성 결함 3건과 taxonomy 단절. P04의 단조성/zero-truncation 수정, P05의 ROI + 유지 카운트 일치, P03/P05 세그먼트 vocabulary 통일을 고객 노출 전 완료해야 함.

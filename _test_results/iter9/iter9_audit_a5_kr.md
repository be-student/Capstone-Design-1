# A5 — 추천(Recommendations) / CLV / 유지 캠페인(Retention Campaign)

감사자 입장: 사전 컨텍스트 없는 독립 리뷰어. 인용된 모든 숫자는 `C:\Users\yoonc\Capstone-Design-1\_test_results\` 아래 PNG 스크린샷과 매칭되는 `page_data/*.md` 덤프에서 직접 가져옴.

---

## Page 09 — Recommendations

### 화면에 보이는 KPI
- 상단 KPI strip: Total Recommendations **20,000** | Avg Expected Uplift **6.36%** | Top Action Type **No Action** | High Priority **16,106**
- Cost-Benefit strip (페이지 중간): Total Campaign Cost **1,211,055 KRW** | Est. Revenue Saved **10,893,463 KRW** | Overall ROI **9.0x** | Avg Expected Uplift **10.88%**
- Recommendation Distribution: no_action **16,602 (83%)**, coupon **3,398 (17%)**.
- Avg Uplift by Action: coupon **16.12%**, no_action **4.36%**.
- Cost-by-Offer-Type: premium_discount 559,170 / discount_coupon 528,658 / engagement_email 108,831 / loyalty_points 14,396 (합계 1,211,055 ✓).
- ROI by Offer Type: premium_discount **10.1x**, loyalty_points **9.8x**, engagement_email **9.3x**, discount_coupon **7.8x**.
- Priority Score Distribution 평균 ≈ 0.81; High Priority 버킷 = 16,106 / 20,000 = 80.5%.

### 잘못된 / 신뢰성 부족
- **같은 라벨 다른 값을 가진 두 개의 "Avg Expected Uplift" KPI**: 6.36% (상단, 20,000 고객 전체에 대해 계산) vs 10.88% (중간, 처치된 3,398명에 대해 계산). 동일 표현, 각주 없음, 범위 한정자 없음 — 분석가가 source code를 읽지 않고는 어느 게 어느 건지 알 수 없음. 이 페이지의 핵심 money-domain 신뢰 실패.
- **High Priority 16,106 vs coupon 받는 고객 3,398** → **12,708명의 "high priority" 고객이 `no_action`을 받음**. 페이지가 "priority"와 "treated"를 조정하지 않음; priority가 개입을 유도해야 한다면, priority 큐의 ~79%가 명시적으로 무시되고 있고 그게 침묵 상태.
- **Overall ROI 9.0x**가 같은 캠페인에 대해 Page 05의 3.5x와 Page 12의 3.8x와 불일치. Cost 1,211,055와 Revenue Saved 10,893,463으로 이 페이지의 산술(10,893,463 / 1,211,055 = 8.99)은 내부 일관됨, 즉 격차는 정의 문제(처치된 부분집합 vs 전체 예산 vs 전체 모집단)이지 산술 버그가 아님 — 그런데 그 자격 표시 없이 보여짐.
- ROI 9.0x는 Page 12의 "Revenue Saved 192M / Budget 50M" 옆에 있을 때 의심스러움 → 이 페이지가 50M 예산 envelope이 아니라 실제 쿠폰에 사용된 1.2M만 카운트한다는 걸 시사. 방어 가능한 분석적 선택이지만 카드에 문서화되지 않음.
- Avg Uplift by Action: no_action이 **4.36%** uplift 표시 — 처치를 받지 않은 고객의 uplift는 정의상 0이어야 함; 이는 미처치 모집단에 대한 "처치되었다면 예측되는 uplift", 또 라벨 없음.

### 누락된 항목
- 모델/버전, 학습 타임스탬프, 데이터 freshness, 코호트-as-of 일자 부재.
- "High Priority" 임계값 정의 부재 (0.81 컷이 암묵적).
- 쿠폰 자격에 대한 churn-probability 하한 표시 부재.
- Uplift에 CI / variance 밴드 부재.
- "High Priority" 카운트와 "treated" 카운트 사이의 조정 패널 부재.

---

## Page 10 — CLV Prediction

### 화면에 보이는 KPI
- 상단 KPI strip: Total CLV **57,936,514,970 KRW** | Average CLV **2,896,826 KRW** | Median CLV **1,701,727 KRW** | CLV Std Dev **3,575,497 KRW**.
- 57,936,514,970 / 20,000 = 2,896,825.7 ≈ Average CLV ✓.
- CLV by Segment (Mean | Total):
  - high_value_persuadable 8,563,357 | **17,126,714**
  - high_value_sure_thing 8,416,210 | 33,639,590,866
  - high_value_lost_cause 4,153,154 | **4,153,154**
  - mid_value_sure_thing 2,443,032 | 16,942,427,813
  - mid_value_persuadable 1,756,438 | 1,097,773,503
  - low_value_sure_thing 1,171,391 | 2,512,633,819
  - sleeping_dog 722,502 | 2,546,096,653
  - low_value_persuadable 424,653 | 1,176,712,448
- Percentile: P10 66,322 / P25 973,223 / P50 1,701,727 / P75 3,075,712 / P90 6,767,562 / P95 13,032,750 / P99 14,716,426.
- CLV Tier Distribution: Platinum/Gold/Silver/Bronze 각 25% (통계적 사분위, 비즈니스 tier 아님).

### 잘못된 / 신뢰성 부족
- **헤드라인에 가려진 작은 세그먼트들**:
  - high_value_persuadable: Total 17,126,714 / Mean 8,563,357 → **n = 2명**.
  - high_value_lost_cause: Total 4,153,154 = Mean 4,153,154 → **n = 1명**.
  - 두 세그먼트 합쳐 3명을 보유, 33.6B / 16.9B 세그먼트 옆에 카운트 컬럼이 강조되지 않은 채 제시됨. "Mean CLV by Segment" 막대 차트가 n=1과 n=4,000 세그먼트를 시각적으로 동등하게 처리 — 평균으로 정렬하거나 예산 짜는 사람이 있다면 buyer-trust 위험.
- **CLV vs Churn Risk 산점도 x축이 대략 −0.5에서 1.5 범위** — 확률 축은 [0, 1]이어야 함. matplotlib auto-padding이 고객 뷰까지 누출되었거나, 기저 churn_probability 컬럼에 [0,1] 밖 값이 있거나. 어느 쪽이든 페이지가 검증되지 않은 것처럼 보이게 만듦.
- **CLV Tier Distribution이 정확히 25/25/25/25** — "Tier Classification"으로 라벨링되었지만 예측 CLV 컬럼에 대한 단순 사분위 컷일 뿐. SaaS 구매자가 "Platinum/Gold/Silver/Bronze"를 읽으면 비즈니스 정의 (예: 매출, 보유 기간)를 가정하고 오해함.
- 여기 사용된 8 uplift 세그먼트 taxonomy는 Page 03 / Page 12의 Section 1에서 사용된 행동 세그먼트 taxonomy와 일치하지 않음; 대시보드에 두 평행 고객 ontology가 있고 그 사이 매핑이 없음.

### 누락된 항목
- by-segment 표에 **세그먼트 카운트 컬럼 부재** (정확히 n=1과 n=2를 노출시켜줄 그 컬럼).
- 모델 버전, 학습 일자, 산점도를 공급하는 churn-prob 모델, 데이터 freshness stamp 부재.
- Percentile 또는 평균에 CI 부재.
- n=1/n=2 세그먼트에 outlier 플래그 부재.
- Top/Bottom 10 고객 표가 churn_probability 컬럼에 `nan` 표시 (PNG에서 확인 가능) — 침묵의 NaN 전파.

---

## Page 12 — CLV & Retention Campaign

### 화면에 보이는 KPI
- Section 1 (CLV): Total CLV **57,936,514,970 KRW** | Avg CLV **2,896,826 KRW** | At-Risk CLV **2,997,471,916 KRW** | At-Risk CLV % **5.2%**.
- Section 2 (Uplift): Avg Uplift **0.0434** | Max Uplift **0.6874** | Treatable Customers **16,317 (81.6%)**.
- Section 3 (Budget): Budget Allocated **50,000,000 KRW** | Revenue Saved **192,155,554 KRW** | Customers Retained **122.29548658078494** | Overall ROI **3.8x**.
- ROI by Segment: high_value_persuadable 8.0x, high_value_sure_thing 5.1x, mid_value_persuadable 4.1x, low_value_persuadable 3.8x, mid_value_sure_thing 3.5x, low_value_sure_thing 3.1x, sleeping_dog 0.0x, high_value_lost_cause 0.0x.
- Expected Revenue Saved by Segment 합 = 192,155,553 ≈ 헤드라인 192,155,554 (1 KRW 반올림 ✓).

### 잘못된 / 신뢰성 부족
- **`Customers Retained = 122.29548658078494`** — 고객용 KPI 카드에 흘러나온 IEEE-754 float 그대로 (소수점 14자리). 한 줄짜리 format helper 버그이며 money domain 전체에서 가장 손상이 큰 시각적 결함 — "테스트 안 됨"을 외침.
- **"Overall ROI 3.8x"가 low_value_persuadable의 세그먼트 ROI 3.8x와 비트 단위로 동일.** "Overall" 타일이 우연히 그 세그먼트의 ROI를 끌어왔거나, 각주가 필요한 우연.
- **같은 캠페인에 대해 제품 전체에서 세 가지 다른 "Overall ROI" 값**: Page 05 = 3.5x, Page 09 = 9.0x, Page 12 = 3.8x. 세 값 어디에도 정의 표시 없음.
- **조정되어야 하지만 그렇지 않은 두 개의 "uplift" 헤드라인 지표**:
  - Page 09 Avg Expected Uplift = 6.36% (≈ 0.0636).
  - Page 12 Avg Uplift = 0.0434.
  둘 다 "같은 20k 모집단에 대한 평균 uplift"로 광고되며 범위 한정자 없음; 단위 정규화 후에도 0.0434 ≠ 0.0636.
- **단일 페이지 내 taxonomy 혼용**: Section 1 ("Customer Lifetime Value Overview") 차트는 행동 세그먼트 (vip_loyal, dormant, …) 사용; Sections 2-4는 uplift 세그먼트 (high/mid/low_value_persuadable, sure_thing, lost_cause, sleeping_dog) 사용. 독자가 섹션 간에 고객을 추적할 수 없음.
- Section 4의 Campaign Effectiveness Radar가 다각형 외곽선만 렌더, 수치 스케일 라벨 없음 — 시각적으로는 존재하지만 분석적으로 불투명.
- Sleeping_dog과 high_value_lost_cause 둘 다 ROI 0.0x와 Cost/Retention 0 표시 — 즉 지출에서 제외됨, 그런데 ROI 표가 그들을 별도 "Excluded" 패널이 아니라 그대로 하단에 나열, "0 벌었음"과 "의도적 미처치"를 섞음.

### 누락된 항목
- "Overall ROI" 정의 각주 부재 (분모: 지출? 예산? at-risk CLV?).
- "Avg Uplift" 정의 각주 부재 (모집단: 전체 20k? treatable 16,317? treated 3,398?).
- Customers Retained에 정수 포매터 없음, CI 없음 (122 자체가 모델 도출 기댓값이지 카운트가 아님).
- 페이지에 데이터 freshness / 모델 버전 부재.
- Pages 05 / 09 / 11을 가리키는 조정 패널 부재.

---

## 페이지 간 money KPI 표

| KPI | P05 (LP) | P09 (Reco) | P10 (CLV) | P12 (Retention) |
|---|---|---|---|---|
| Total CLV | — | — | 57,936,514,970 KRW | 57,936,514,970 KRW |
| Avg CLV | — | — | 2,896,826 KRW | 2,896,826 KRW |
| Median CLV | — | — | 1,701,727 KRW | — |
| At-Risk CLV | — | — | — | 2,997,471,916 KRW (5.2%) |
| Total Recommendations | — | 20,000 | — | — |
| High Priority count | — | 16,106 | — | — |
| Coupons issued | — | 3,398 | — | — |
| Treatable Customers | — | — | — | 16,317 (81.6%) |
| Avg Expected Uplift / Avg Uplift | — | **6.36%** (상단) / **10.88%** (중간) | — | **0.0434** |
| Max Uplift | — | — | — | 0.6874 |
| Budget Allocated | 50,000,000 KRW | — | — | 50,000,000 KRW |
| Total Campaign Cost (지출) | — | 1,211,055 KRW | — | — |
| Revenue Saved | ~192,155,551 KRW | 10,893,463 KRW | — | 192,155,554 KRW |
| Overall ROI | **3.5x** | **9.0x** | — | **3.8x** |
| Customers Retained | 118 | — | — | **122.29548658078494** |

세 가지 다른 "Overall ROI" 값, 세 가지 다른 "Revenue Saved" 규모 (둘은 "spent" vs "budget" 범위), 두 가지 다른 "Customers Retained" 카운트, 두 가지 다른 "Avg Uplift" 헤드라인 — UI에서 어느 것도 조정되지 않음.

---

## SaaS-readiness 평결 — Money 도메인

**평결: 유료 고객 사용에 대해 NOT READY (준비 안 됨).** 단일 페이지 내 산술은 내부 일관됨 (반올림 범위 내에서 합산 일치, Total ÷ N = Mean, 세그먼트 매출이 헤드라인 합계로 합산), 그러나 *페이지 간 money 스토리가 일관되지 않으며* 적어도 하나의 cosmetic 버그 (`122.29548658078494`)가 제품을 베타처럼 보이게 만듦. 재무 또는 성장 부서 구매자가 Page 05 → 09 → 12를 클릭하면 세 가지 ROI와 두 가지 uplift 값을 보고 PO에 사인하지 않을 것.

**Top 3 buyer-trust 블로커:**
1. **하나의 캠페인에 세 가지 "Overall ROI" 값 (3.5x / 9.0x / 3.8x), 어디에도 정의 각주 없음.** 같은 라벨, 다른 분모, 글로사리 없음. 재무 리뷰어가 대시보드를 거절할 #1 이유.
2. **`Customers Retained = 122.29548658078494`**, Page 12. 고객 카운트 카드의 14자리 float은 full QA 재리뷰를 트리거하고 인접한 모든 숫자에 대한 신뢰를 지우는 종류의 artifact.
3. **작은 세그먼트가 헤드라인 세그먼트로 위장.** Page 10이 high_value_persuadable (n=2)와 high_value_lost_cause (n=1)를 n=4,000+ 세그먼트와 같은 축에 표시, 카운트 컬럼이 강조되지 않음 — 그리고 Page 09는 16,106 "High Priority"인데 쿠폰은 3,398개뿐, 12,708명의 high-priority 고객이 침묵 속에 `no_action`. 어느 차트를 읽어도 SaaS 구매자가 잘못된 상업적 결정에 도달함.

가산점 사항 (각각 <1일에 수정 가능): Page 10 churn 산점도 x축의 음수 확률, 같은 라벨의 Page 09 이중 "Avg Expected Uplift" 카드, Page 12 내부의 behavioral-vs-uplift taxonomy 혼용.

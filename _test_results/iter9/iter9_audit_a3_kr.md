# A3 — A/B 테스트 / 생존(Survival) / 업리프트(Uplift)

감사자: 독립 (사전 컨텍스트 없음). 출처: `_test_results/dashboard_pages/`의 PNG 스크린샷 + `_test_results/page_data/`의 MD 데이터 덤프. 세 페이지 모두 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated." 배너 표시.

---

## Page 06 — A/B Testing

### 화면에 보이는 KPI (그대로)
- Total Experiments: **0**
- Significant Results: **0**
- Best Experiment: **N/A**
- Avg Lift: **0.0%**
- Required Sample Size (per group): **906**
- Total Participants Needed: **1,812**
- Expected Duration (days): **19**
- Power 입력값: Baseline Churn 0.20, MDE 0.05, alpha 0.05, power 0.8
- MDE Sensitivity 표: MDE 1% → 24,441/group (48,882 total); 2% → 6,059/12,118; 3% → 2,629/5,258; 5% → 906/1,812; 8% → 329/658; 10% → 199/398; 15% → 70/152

### 잘못된 부분
- **Best Experiment "N/A"와 Avg Lift "0.0%"의 framing 모순**: 0.0%는 *측정된* 수치를 소수점 한 자리로 표시한 것인데, 측정할 실험이 0개. "—" 또는 숨김으로 표시해야 함, 숫자 KPI로 포맷팅하면 안 됨.
- **MDE 1%는 그룹당 24,441명 = 총 48,882명** 필요한데 이는 20,000 고객 풀을 초과. 페이지에 "현재 풀 크기에서 실현 불가" 가드레일 없음.

### 신뢰성 부족
- 헤드라인 KPI 4개 (0/0/N/A/0.0%)가 "아직 실험 없음" 빈 상태가 아니라 망가진 모델처럼 읽힘. 빈 상태 일러스트, CTA ("Launch first experiment"), 0의 의미를 설명하는 문구 없음.
- Power 계산기는 20,000 고객 풀을 무한으로 취급 — "고객이 충분한가?"를 묻지 않음. MDE 5%는 1,812명 필요, MDE 1%는 48,882명 필요(즉 풀의 2.4배)인데 경고 표시 없음.

### 누락된 항목
- `Required Sample Size = 1,812`와 20,000 고객 풀 사이의 조정 부재 ("11배 여유" 같은 헤드룸 인디케이터 없음).
- 실험 감사 로그 / 과거 실험 이력 부재.
- 다중 비교 보정 (Bonferroni / BH FDR), 양측/단측 토글, 분산 조정(CUPED) 옵션 부재.
- SRM (sample-ratio mismatch) 점검 부재, 최소 실행 시간 / 주간 계절성 가이드 부재.
- 세그멘테이션 토글 부재 (power calc이 글로벌 전용).
- 실세계 lift 정상 범위 disclaimer (보통 5–15%) 부재.

---

## Page 07 — Survival Analysis

### 화면에 보이는 KPI (그대로)
- Total Customers: **20,000**
- Events Observed (Churn): **5,717**
- Event Rate: **28.59%**
- Median Duration: **309 days**
- Avg Survival by Segment: high_value_sure_thing 97.68%, mid_value_sure_thing 86.98%, low_value_sure_thing 73.04%, high_value_persuadable 39.84%, mid_value_persuadable 39.78%, sleeping_dog 38.81%, high_value_lost_cause 37.23%, low_value_persuadable 22.28%
- Daily Hazard by Segment: dormant 0.00254, new_customer 0.00166, bargain_hunter 0.00120, explorer 0.00099, regular_loyal 0.00062, vip_loyal 0.00023
- Event Rate by Segment: high_value_lost_cause **100.0%**, high_value_persuadable **100.0%**, low_value_persuadable **100.0%**, mid_value_persuadable **100.0%**, sleeping_dog **65.8%**, high_value_sure_thing **0.0%**, low_value_sure_thing **0.0%**, mid_value_sure_thing **0.0%** (총 8 세그먼트 — "몇 개 세그먼트인가?"의 답 → **8**)
- 모델 config: penalizer 0.01, l1_ratio 0, alpha 0.05

### 잘못된 부분
- **Events Observed = 5,717**은 Overview 페이지에서 "High Risk (>50%) count = 5,717"로 보고된 것과 동일한 숫자 (MD의 분석가 메모 참조). 이는 "예측된 고위험 카운트"가 "관측된 이벤트"로 잘못 라벨링된 것 — 범주 오류: 예측은 결과가 아님.
- **Event Rate by Segment가 binary**: 8개 세그먼트 중 7개가 정확히 0.0% 또는 100.0% 보고 (`sleeping_dog` 65.8%만 패턴 깸). 이는 동의어 반복(tautological) 라벨링 artifact — 세그먼트 정의가 결과 라벨을 누설하는 것 — 신뢰할 수 있는 production 신호가 아님.
- **한 페이지에 두 가지 세그먼트 taxonomy**: Avg Survival은 8버킷 uplift taxonomy (`high/mid/low × value × persuadable/sure_thing/lost_cause/sleeping_dog`) 사용, Daily Hazard는 6버킷 행동 taxonomy (`vip_loyal / regular_loyal / bargain_hunter / explorer / dormant / new_customer`) 사용. 차트들이 직접 비교 불가능.
- **350일 horizon에서 Median 309일** = right-censoring artifact; median이 관측 윈도우에 부딪히고 있음. 주석 없음, "horizon-bounded" 경고 없음, 더 긴 follow-up으로 재실행 없음.

### 신뢰성 부족
- KM 곡선이 95% 신뢰 구간 밴드 없이 표시되고 아래 "at-risk" 표 없어 곡선 검증 불가.
- Customer Duration Distribution이 350-day bin에서 `mid_value_sure_thing`의 거대한 spike를 보여줌 — 윈도우 경계에서의 무거운 right-censoring 확인; 따라서 "sure_thing" 세그먼트의 생존 확률은 구조적으로 낙관적.
- 28.59% event rate는 그럴듯하지만, 5,717 = 예측된 고위험 카운트 충돌을 고려하면 *관측된* 비율로 신뢰 불가.

### 누락된 항목
- 세그먼트 비교를 위한 log-rank / Cox PH 검정 부재.
- KM 곡선 또는 median duration에 신뢰 구간 부재.
- 곡선 아래 at-risk 숫자 부재.
- Censoring 주석 / 처리 부재.
- Restricted mean survival time (RMST) 부재, hazard ratio 표 부재.
- "이벤트"가 무엇으로 카운트되는지 정의 부재 (churn 라벨인지, last_purchase>X일인지, risk_score>0.5인지?).

---

## Page 11 — Uplift Modeling

### 화면에 보이는 KPI (그대로)
- Avg Uplift Score: **0.0434**
- Avg Treatment Effect: **0.0434**
- Persuadable Customers: **16,317**
- Sleeping Dogs: **3,683**
- Avg Uplift by Segment 표: persuadable 0.1902 (count 600 → 표 실제로는 lost_cause 0.0258 count 600, persuadable 0.1902 count 2,708, sleeping_dog -0.1097 count 3,683, sure_thing 0.0560 count 12,929; Persuadable% 컬럼은 100% / 0% / 0% / 100%)
- Customer Response Classification 파이: **Persuadable 81.6% / Lost Cause 18.4%**
- Top-10 Persuadable 고객 표 (모든 행에서 uplift_score == treatment_effect 4 dp까지 일치, 예: 0.6874/0.6874, 0.6343/0.6343, 0.64/0.64, …)

### 잘못된 부분
- **Avg Uplift Score (0.0434) == Avg Treatment Effect (0.0434)** 4자리 소수점까지 일치. "Top 10 Persuadable" 표는 모든 단일 행에서 uplift_score == treatment_effect 4 dp까지 일치 (0.6874=0.6874, 0.6343=0.6343, 0.64=0.64, 0.6386=0.6386, …). 이들은 두 지표가 아님 — 한 변수를 두 라벨로 두 번 렌더링한 것. 동일 이유로 두 분포 히스토그램 ("Distribution of Uplift Scores" vs "Distribution of Treatment Effects")이 시각적으로 동일.
- **16,317 + 3,683 = 20,000** → 헤드라인 KPI에서 전체 고객 베이스를 4개 사분면 중 2개로만 강제 분류. 아래 막대/표에는 4개 모두 표시 (lost_cause 600, persuadable 2,708, sleeping_dog 3,683, sure_thing 12,929 = 20,000) — 즉 데이터에는 4개 사분면이 있는데 헤드라인이 2개로 collapse. 헤드라인의 16,317은 어느 단일 세그먼트 카운트와도 일치하지 않음: `persuadable (2,708) + sure_thing (12,929) + lost_cause (600) + ...`도 아님. 실제로 2,708 + 12,929 + 600 = 16,237이지 16,317이 아님. 16,317이라는 숫자는 같은 페이지에 표시된 **어떤** 세그먼트 카운트와도 일치하지 않음.
- **한 페이지 내 네이밍 불일치**: 헤드라인은 "Persuadable / Sleeping Dogs"; 파이 차트 범례는 "Persuadable / Lost Cause"; 세그먼트 표는 4 라벨 모두 표시 {lost_cause, persuadable, sleeping_dog, sure_thing}. 같은 페이지의 세 섹션에 세 가지 다른 vocabulary.
- **Page 11은 4버킷 uplift taxonomy 사용**, 한편 (MD의 분석가 메모에 따르면) Pages 05/10은 8버킷 value × uplift taxonomy 사용. 페이지 간 taxonomy drift.
- **Sleeping_dog Avg Uplift = -0.1097 (음수)** — 이 고객들은 처치(treatment)에 의해 *해를 입음*. 어떤 쿠폰/유지 캠페인에서도 이들을 제외해야 한다는 인라인 가드레일 없음.

### 신뢰성 부족
- Persuadable% 컬럼이 lost_cause와 persuadable에 `100.0%`, sleeping_dog와 sure_thing에 `0.0%` 보고 — 또 다른 binary 0/100 세그먼트 누설 패턴 (Page 07의 Event Rate by Segment를 미러링).
- `persuadable`의 "Avg Uplift" 0.1902, sample size 2,708인데 CI 표시 없음.
- Distribution of Uplift Scores와 Distribution of Treatment Effects가 마치 독립 인사이트인 것처럼 나란히 제시 — 같은 그림.

### 누락된 항목
- Uplift score에 CI / 표준오차 부재.
- Qini curve 또는 uplift gain curve (uplift 검증의 정통 차트) 부재.
- 처치/대조 split 공개 (arm별 샘플 크기, 무작위화 점검) 부재.
- Barrier/처치 비용 ROI 통합 부재 (표가 모든 top-10 행에 `selected_barrier = t_barrier` 표시하지만 비용 또는 expected value 없음).
- Negative-uplift 고객(sleeping_dogs) 제외 가드레일 부재.
- 헤드라인 `Persuadable=16,317`과 표 `persuadable count=2,708`의 조정 부재.

---

## 페이지 간 일관성 점검 (특히 5,717 / 16,317 / 3,683 / 20,000 숫자에 대해)

| 숫자 | 등장 위치 | 주장된 의미 | 실제 |
|---:|---|---|---|
| 20,000 | Page 06 배너, Page 07 Total Customers, Page 11 (16,317+3,683) | 고객 풀 크기 | 일관됨 — n. |
| 5,717 | Page 07 "Events Observed (Churn)"; MD 메모에 따르면 Page 01도 같은 값을 "High Risk (>50%) count"로 보고 | Survival 페이지는 관측 이벤트라 부르고; Overview는 예측된 고위험 카운트라 부름 | 두 개념은 다름. 사후 관측된 이탈이거나 OR 예측 이탈 확률 > 0.5 고객 카운트. 둘 다일 수는 없음. 충돌은 Page 07이 예측 기반 라벨을 관측 이벤트처럼 다루고 있음을 강력 시사 — 그렇다면 전체 Kaplan-Meier 곡선이 무효화됨. |
| 16,317 | Page 11 헤드라인 "Persuadable Customers" | persuadable 고객 수 | Page 11 자체의 세그먼트 표가 persuadable count = 2,708 (16,317 아님) 표시. 16,317은 표의 어떤 부분합과도 일치하지 않음 (가장 가까운 것: persuadable + sure_thing = 15,637; persuadable + sure_thing + lost_cause = 16,237). 헤드라인 수치가 같은 페이지의 supporting 표와 조정되지 않음. |
| 3,683 | Page 11 헤드라인 "Sleeping Dogs" | sleeping_dogs 카운트 | 세그먼트 표 sleeping_dog count = 3,683과 일치. 내부 일관됨. |
| 16,317 + 3,683 = 20,000 | Page 11 헤드라인 산술 | "전체 베이스가 Persuadable 또는 Sleeping Dog로 분류됨" | 같은 페이지의 표가 4 세그먼트로 20,000 합산 (sure_thing 12,929 + persuadable 2,708 + sleeping_dog 3,683 + lost_cause 600)임에 모순. 헤드라인이 4 → 2로 잘못 collapse. |
| Avg Uplift = Avg Treatment Effect = 0.0434 | Page 11 헤드라인 + Top-10 표의 모든 행 | 두 개의 별개 지표 | 한 변수, 두 라벨. 거짓 등가성을 corroboration처럼 제시. |
| 0% / 100% 세그먼트 패턴 | Page 07 Event Rate by Segment (8 중 7개가 0 또는 100), Page 11 Persuadable% 컬럼 (4 중 4개가 0 또는 100) | 실제 세그먼트별 비율 | 동의어 반복 — 세그먼트 정의가 결과 라벨을 누설. Production 신호 아님. |
| 세그먼트 taxonomy | Page 03 (6-behavioral); Page 07 (Avg Survival에 8-uplift + Daily Hazard에 6-behavioral 혼용); Page 10/05 (8-uplift); Page 11 (4-quadrant uplift) | 문서화된 crosswalk가 있는 하나의 정통 taxonomy여야 함 | 대시보드 전반에 4가지 다른 taxonomy 등장, crosswalk 없음. Page 07은 한 화면에 두 가지 사용. |

---

## SaaS-readiness 평결

**평결: SaaS / 유료 고객에 대해 NOT READY (준비 안 됨).** 세 분석 페이지가 수치적으로 잘못되고, 내부적으로 모순되며, 개념적으로 혼란스러운 콘텐츠를 "Deploy" 버튼 바로 위에 배치. Page 11의 단일 스크린샷 (헤드라인 Persuadable=16,317, 바로 아래 표 persuadable count=2,708; "Avg Uplift"가 모든 행에서 "Avg Treatment Effect"와 4자리 소수점까지 동일)만으로 엔터프라이즈 신뢰를 잃기에 충분. Page 07은 예측 카운트(5,717)를 관측 이벤트로 재사용 — 이는 통계적으로 무효이지 단순 cosmetic 결함이 아님. Page 06은 0/0/N/A/0.0%를 빈 상태가 아니라 KPI로 앞세움.

**Top 3 블로커:**
1. **Page 11: Avg Uplift Score == Avg Treatment Effect (0.0434), 모든 고객 행에서 uplift_score == treatment_effect.** 모델이 문자 그대로 한 값을 두 번 반환하거나 (그러면 하나 제거), 두 계산이 잘못 alias되었거나 (그러면 버그 수정). 또한 헤드라인 "16,317 Persuadable + 3,683 Sleeping Dogs = 20,000"이 같은 페이지의 세그먼트 표 (persuadable=2,708, 4개 세그먼트 합 20,000)와 모순, 같은 페이지의 파이 차트 ("Lost Cause")와 다른 vocabulary 사용.
2. **Page 07: Events Observed = 5,717이 Overview 페이지의 "High Risk count = 5,717"과 충돌.** 예측이 관측처럼 처리되고 있음; 이는 Kaplan-Meier 곡선, hazard rate, median 309 d, 28.59% event rate를 무효화. 더해서 `Event Rate by Segment`가 8 중 7개에서 binary 0%/100% — 동의어 반복 라벨 누설 패턴, Page 11의 Persuadable% 컬럼에도 동일 등장. 두 가지 세그먼트 taxonomy (8-uplift, 6-behavioral)가 한 페이지에 혼용.
3. **Page 06: 0/0/N/A/0.0% 헤드라인 KPI에 빈 상태 처리 없음**, power 계산기가 20,000 고객 풀보다 큰 샘플 크기 (MDE 1%에서 그룹당 24,441)를 거리낌없이 추천. 망가진 모델처럼 읽히지 "아직 실험 없음"이 아님; 실제 고객 베이스에 대한 실현 가능성 가드 제공 안 함.

---

## 5줄 요약

1. Page 11 치명적 결함: Avg Uplift Score == Avg Treatment Effect (둘 다 0.0434), 모든 고객 행에서 동일 — 한 변수가 두 지표로 제시.
2. Page 11 산술 깨짐: 헤드라인 "Persuadable 16,317 + Sleeping Dogs 3,683 = 20,000"이 같은 페이지의 4-세그먼트 표 (persuadable = 2,708)와 모순, 파이 범례는 두 번째 그룹을 "Lost Cause"로 잘못 라벨.
3. Page 07 치명적 결함: Events Observed = 5,717이 Page 01의 "High Risk count"와 같은 숫자 — 예측이 관측 이벤트로 기록되어 KM/hazard 분석 무효화.
4. Page 07 보조: 8개 세그먼트 중 7개에서 Event Rate binary 0%/100% (동의어 반복 라벨 누설); 두 세그먼트 taxonomy (8-uplift + 6-behavioral)가 한 페이지에 혼용; 350일 horizon의 median 309일은 주석 없는 censoring artifact.
5. Page 06: 0/0/N/A/0.0%가 빈 상태 대신 KPI로 framing; power 계산이 20,000 풀에 대해 최대 48,882명 추천, 실현 가능성 가드 없음. 평결: NOT SaaS-ready.

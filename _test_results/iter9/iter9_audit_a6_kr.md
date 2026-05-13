# A6 — 실시간 채점(Real-Time Scoring, 3 탭)

출처 artifact:
- Tab a (Live): `_test_results/dashboard_pages/13_realtime_scoring_a_live.png`
- Tab b (Offers): `_test_results/dashboard_pages/13_realtime_scoring_b_offers.png`
- Tab c (Monitoring): `_test_results/dashboard_pages/13_realtime_scoring_c_monitoring.png`
- 통합 데이터 MD: `_test_results/page_data/13_realtime_scoring.md`

페이지 배너 (모든 탭): "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated." + "Redis: Connected" + "Recommended Offer: no_action".

---

## Tab a — Live Scoring Status

### 화면에 보이는 KPI
- **Service Health**: Redis Connected; Request Stream = **0**; Response Stream = **0**; Consumer Group = `scoring_consumers`.
- **Recent Scoring History**: Total Scores = **200**; Avg Churn Prob = **27.30%**; High/Critical Risk = **17**; Primary Model = `ensemble`.
- 차트: Scoring Requests per Minute (x축 Oct 15 2024); Response Latency & Error Rate dual-axis (Oct 15→Oct 16 2024, latency 15–35 ms, error 0–5%); 200 score의 Churn Probability Distribution 히스토그램; Risk Level Distribution 막대 (low/medium/high/critical). Detailed Scoring Log expander (Latest 50).

### 잘못된 / 모순
- **Request Stream = 0 AND Response Stream = 0 vs Total Scores = 200** — 직접 KPI 모순. 두 큐가 모두 비었으면 SaaS SRE는 컨슈머가 죽었다고 결론지을 것; 그런데 200 score가 흘러간 것으로 보임. 라벨이 다른 것을 의미하는데 (큐 깊이 vs 누적 처리량) UI가 disambiguate를 실패했거나, 진짜 스트림이 죽었고 200이 stale fixture data이거나.
- **Time anchor가 stale**: throughput/latency 차트가 "Real-time"으로 **Oct 15–16, 2024**를 plot. *Real-Time Scoring*이라는 제목의 페이지에서, 페이지 다른 곳에서 사용되는 시스템 시계 (May 10, 2026, Tab c 참조) 대비 ~19개월 뒤처짐.
- **High/Critical Risk = 17 / 200 = 8.5%**가 절대 숫자로 제시됨, 비율 없음, 추세 없음, SLA 임계값 없음.

### 신뢰성 부족
- **"200" Total Scores**에 시간 윈도우 라벨 없음 — lifetime인지, 지난 한 시간인지, 마지막 refresh인지? 차트 x축은 ~24h 걸쳐있는데 KPI는 일자 없음.
- Latency 차트 축 15–35 ms에 SLO 선 없음, p50/p95/p99 분리 없음 — error rate 축 0–5%에 alert 임계값 없음.
- 200 churn 확률의 히스토그램은 20,000 고객 모집단에 대한 분포 결론을 도출하기에 너무 작은 샘플.

### 누락된 항목
- p50/p95/p99 latency 분리 없음, QPS 카운터 없음, error budget burn 없음, scored-vs-failed 분리 없음, 표시된 score에 묶인 모델 버전 없음, 마지막 채점 타임스탬프 없음, consumer lag 없음 ("stream depth = 0"만).
- Total Scores 200에서 20,000 고객 모집단 분모로의 링크 없음.
- Service Health 카드에 "as of" 타임스탬프 없음.

---

## Tab b — Retention Offer Recommendations

### 화면에 보이는 KPI
- **Total Offers = 44**; **Total Cost = 1,196,659 KRW**; **Expected Revenue Saved = 10,752,341 KRW**; **Expected ROI = 8.0x**.
- 필터: Risk Level (critical, high, medium), Segment (bargain_hunter, dormant, explorer, new_customer, regular_loyal, vip_loyal), Offer Type (discount_coupon, engagement_email, loyalty_points, premium_discount).
- 차트: Offer Type Distribution 파이; Average Expected Uplift by Segment 막대; Cost vs Expected Revenue Saved by Segment grouped bar; Churn Probability vs Expected Uplift 산점도.
- Detailed Offer Recommendations 표 (priority_score, customer_id, segment, risk_level, churn_probability, offer_type, offer_detail, expected_uplift, estimated_cost…).
- **Quick Recommendation Lookup** 고객 138000: Recommendation = **no_action**; Expected Uplift = **1.46%**; Priority Score = **1.00**; 배너 "Recommended Offer: no_action".

### 잘못된 / 모순
- **카드의 ROI 산술이 틀림.** 10,752,341 / 1,196,659 = **8.985x** (≈ 9.0x). 카드는 **8.0x** 표시 — ~11% 차이. 숫자가 truncate된 것이거나 (반올림 아님), 세 numerator/denominator/ROI 타일 중 하나가 다른 스냅샷에서 온 것이거나.
- **Quick Lookup 카드의 "no_action" + Priority Score 1.00 + Expected Uplift 1.46%는 내부 모순.** Priority가 1.00으로 max라면 추천은 가장 높은-EV offer여야지 "no_action"이 아님. "no_action"이 맞다면 (uplift 1.46% < 비용 임계값), Priority Score 1.00이 잘못 라벨링됨 — 이는 액션 priority가 아니라 churn risk score처럼 동작 중.
- 페이지 레벨 배너 "Recommended Offer: no_action"이 어떤 고객도 선택되지 않은 상태에서도 글로벌하게 렌더링됨, 시스템의 모든 사람에 대한 기본 추천처럼 보이게 만듦.

### 신뢰성 부족
- **44 offers — 분모 부재.** 최근 채점된 200 중 44 = 22%; high/critical 17 중 44 = 259% (불가능, 즉 분모는 "high/critical risk"가 아님); 전체 모집단 20,000 중 44 = 0.22%. 카드는 어느 것도 명시 안 함.
- **Total Cost 1,196,659 KRW vs Budget 50,000,000 KRW** (사이드바) — 예산의 2.4%만 소진. 사용률 인디케이터 없음.
- ROI 8x / 9x가 신뢰 밴드 없이 인용, holdout-vs-counterfactual 증거 없음, 세그먼트별 ROI 분포 없음.
- Average Expected Uplift by Segment 차트는 ~5%까지의 uplift를 보여주는데 lookup 카드는 1.46% — 세그먼트 baseline에 대한 anchor 없음.

### 누락된 항목
- Total Offers 카드에 "44 of N scored" 분모 없음; "as of" 타임스탬프 없음; offer-policy 버전 없음; 표 행에서 고객 레코드로의 링크 없음.
- 추천이 `no_action`일 때 "next best action" 대안 표시 없음, reason code 없음, no_action 선택 시 추정 비용 없음.
- 전환률 / 수락률 피드백 루프 표시 없음 — 대시보드가 어제의 offer가 효과 있었는지 루프를 닫지 않음.

---

## Tab c — Model Monitoring

### 화면에 보이는 KPI
- **Total Drift Checks = 1**; **Red Alerts = 1**; **Yellow Warnings = 0**; **Latest Alert Level = RED**.
- 차트: Drift Alerts Over Time (단일 점, May 10 2026 11:56:50, x축 ~1.5 ms span); PSI Trend (단일 점, "Alert PSI 0.25" 선 표시); KS Statistic Trend (단일 점); Scoring Volume Over Time 막대 차트, ~50 hourly bucket이 모두 값 **4**; Mean Churn Probability Over Time line+band (~50 점, Oct 1–Oct 15 2024); Model Type Usage in Recent Scoring 파이 (ensemble / lightgbm / xgboost). Drift Detection History (Full)와 Monitoring Configuration expander.

### 잘못된 / 모순
- **Total Drift Checks = 1, Latest Alert = RED.** 비교 이력 없이 첫 번째 샘플에서 빨간 알람이 발사됨. "이게 n-of-n로 통계적으로 유의한가" 가드 없음 — SaaS 모니터링 도구가 단일 관측에 RED를 발사하는 것은 false-positive 공장.
- **"Drift Alerts Over Time"이 1.5 ms x축 = 정확히 1 datapoint.** 그것을 "trend"라 부르는 건 오해를 부름; 차트는 n=1일 때 시계열로 렌더링되면 안 됨.
- **PSI Trend / KS Trend = 각 단일 점**, 그런데 차트 창에 multi-week 추적을 시사하는 풀 축이 있음.
- **Tab a와의 시간 앵커 불일치.** Drift 차트 일자 **May 10, 2026 11:56:50** (current); Tab a의 throughput/latency 차트 일자 **Oct 15–16, 2024**; 같은 Tab c의 "Mean Churn Probability Over Time" 또한 일자 **Oct 1 – Oct 15 2024**. 같은 "real-time" surface, 두 시계 ~19개월 차이.
- **Scoring Volume Over Time = ~50 hourly bucket 모두에서 4.** Uniform-4는 합성 placeholder 데이터지 텔레메트리가 아님. 실제 채점 서비스는 Poisson-shape 트래픽을 가짐; 평탄한 상수는 seed를 누설.

### 신뢰성 부족
- KPI가 RED 알람을 트리거한 실제 PSI / KS 값을 surface하지 않음 — 카운트만. On-call 엔지니어가 이 카드로 triage 불가.
- "Latest Alert Level = RED"인데 baseline 기간 없음, feature 이름 없음, reference window 없음 — payload 없는 status.
- Model Type Usage 파이에 시간 윈도우 없음, 묶인 샘플 카운트 없음.

### 누락된 항목
- Alert payload 부재 (어떤 feature가 drift했는지, PSI/KS 값, 넘은 임계값, baseline window, 현재 window).
- Mute / acknowledge 상태 부재, 인시던트 링크 부재, alert age 부재.
- 모델 버전 부재, 학습 일자 부재, drift event에 묶인 last-retrain 타임스탬프 부재.
- 모니터링 freshness에 대한 SLA 부재 ("마지막 drift check가 X분 전에 실행됨").

---

## 탭 간 일관성 점검 (타임스탬프, 모델 버전, 총 카운트)

| Surface | 타임스탬프 앵커 | 샘플 카운트 |
|---|---|---|
| Tab a — Scoring Requests per Minute | Oct 15 2024 | (per-minute trace, ~24h) |
| Tab a — Latency & Error Rate | Oct 15–16 2024 | (per-minute trace, ~24h) |
| Tab a — Total Scores 카드 | undated | **200** |
| Tab b — Total Offers 카드 | undated | **44** |
| Tab c — Drift Alerts Over Time | **May 10 2026 11:56:50** (1.5 ms span) | **1** |
| Tab c — Mean Churn Prob Over Time | Oct 1 – Oct 15 2024 | ~50 hourly 점 |
| Tab c — Scoring Volume Over Time | Oct 1 – Oct 15 2024 | ~50 bucket, **모두 = 4** |

- **하나의 "real-time" 페이지에 두 시계 우주**: Oct 2024 (throughput / latency / mean-churn / volume)와 May 2026 (drift alerts). 둘 다 "current"일 수 없음.
- **카운트 조정 불가**: 최근 200 score (Tab a) vs 44 offer (Tab b) vs 1 drift check (Tab c) vs ~50 bucket × 4 score = 200 (Tab c) — 마지막 것만 Tab a로 다시 묶이고, 그것도 bucket이 uniform-4 placeholder임을 무시할 때만 그러함.
- **모델 버전이 어디에도 고정되지 않음.** Tab a는 "Primary Model: ensemble"이라고 함; Tab c 파이는 ensemble + lightgbm + xgboost 혼용. 200 score, 44 offer, 1 drift check를 묶어주는 버전 문자열 없음, 학습 일자 없음, model-id 없음.

---

## SaaS-readiness 평결 — 실시간 서빙

**평결: NOT production-ready.** 이는 실시간 콘솔로 가장한 데모 surface. 스트림은 0인데 score 카운터는 200, ROI는 헤드라인 카드에서 산술 오류 (8.0x인데 underlying numerator/denominator는 8.99x를 시사), drift "trend"는 1.5 ms 축 위 단일 점, scoring volume은 매 시간 평탄한 상수 4, 세 탭 중 둘이 세 번째와 19개월 떨어진 일자. 고객 lookup 카드는 동시에 Priority 1.00 (max)을 보고하면서 1.46% uplift로 `no_action`을 추천함 — priority score가 액션 EV가 아니라 risk에 잘못 연결되어 있음을 사용자에게 노출하는 단서. 이 페이지의 어떤 것도 on-call SRE가 "지금 채점 서비스가 건강한가, 지난 한 시간 동안 무엇이 바뀌었나?"를 답하게 해주지 못함.

**Top 3 production-readiness 블로커:**
1. **KPI 무결성이 헤드라인 레벨에서 깨짐.** Request/Response Stream = 0인데 Total Scores = 200 (Tab a); Total Offers 44에 분모 없음, Expected ROI 8.0x인데 표시된 numerator/denominator는 8.99x를 시사 (Tab b); Total Drift Checks = 1로 단일 샘플에 RED 발사 (Tab c). 네 타일 그룹 중 셋이 자기 underlying 숫자와 불일치 — 신뢰하기 전에 라벨, 단위, 분모, 반올림을 먼저 수정해야 함.
2. **시간 앵커 비일관성과 합성 텔레메트리.** Throughput/latency Oct 15–16 2024, drift May 10 2026 11:56:50 (1.5 ms span), Scoring Volume = ~50 hourly bucket에서 균일하게 4. 페이지가 한 시계에 commit하고, 모든 카드에 "as of"를 표시하고, placeholder uniform-4 volume을 실제 consumer-group 카운터로 교체해야 모니터링 도구가 됨.
3. **추천 엔진 UX가 자기 모순.** Quick Lookup이 reason code 없이, 대안 offer 없이, 모든 방문자에게 `no_action`을 기본으로 두는 글로벌 배너와 함께 `no_action` + Priority Score 1.00 + Expected Uplift 1.46% 반환. Priority Score가 잘못 정의된 것이거나 (액션 priority가 아니라 churn-risk score) 정책이 깨진 것이거나 — 어느 쪽이든 CRM 운영자가 행동할 수 있는 추천이 아님. 출시 전에 reason code, 선택된 offer의 counterfactual, model+policy 버전 stamp 추가 필요.

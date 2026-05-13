# A4 — 모델 모니터링(Monitoring) / MLflow / 시스템 헬스(System Health)

페이지 08, 14, 15에 대한 독립 감사. 출처: `C:\Users\yoonc\Capstone-Design-1\_test_results\` 아래의 PNG 스크린샷 + page_data MD 덤프.

---

## Page 08 — Model Monitoring

### 화면에 보이는 KPI
- Drift: Total Checks=**1**, Current Status=**RED**, Red Alerts=1, Yellow Alerts=0
- Throughput: Avg Req/min=49.0, Peak=83.3, Avg Latency=19.1 ms, Avg Error Rate=**0.0103 (1.03%)**
- Best model by AUC: ensemble (AUC=0.8866)
- 헤드라인 배너: "No performance degradation detected for ensemble."

### 잘못된 / 모순
- **헤드라인 영역에서 직접 모순:** 배너는 *"No performance degradation detected for ensemble"*인데 바로 다음 KPI 타일은 *Drift Status: **RED***. 같은 viewport에서 사용자에게 정반대 두 가지를 알림.
- **1.5밀리초를 span하는 "Trend" 차트.** 세 가지 drift "Over Time" 차트(Drift Alert Timeline, Mean PSI Over Time, Mean KS Statistic Over Time) 모두 x축이 2026년 5월 10일 `11:56:50.282 → 11:56:50.2835` — **~1.5 ms** 윈도우. Total Checks=1이므로, 단일 관측을 추세선으로 그린 것.
- **Training Run History x축이 ~0.1ms 폭** (`11:56:50.2892 → 11:56:50.2893`). 함수 호출 3번을 종단(longitudinal) 시리즈로 plot.
- **같은 페이지에 19개월 시간 앵커 split.** Drift 차트는 2026년 5월 10일 stamp, Throughput / Latency / Error Rate 차트는 **2024년 10월 15–16일** stamp. 같은 모니터링 페이지, 두 다른 epoch.
- **Error rate 1.03%는 표준 SaaS SLO 목표(<0.1%)의 10배** — 그런데 페이지에 이와 연결된 알림 표시 없음.

### 신뢰성 부족
- n=1 샘플에서 도출된 Drift "RED"는 통계적으로 무의미; PSI/KS 임계값(0.1/0.25, 0.05)은 단일 점에서 평가 불가.
- 24시간 Oct 2024 축 위의 Throughput/latency/error 차트가 오늘 날짜의 KPI 타일의 출처일 수 없음; provenance 미정의.
- Page 07의 Kaplan–Meier 생존 곡선이 여기에 그대로 복제됨 (scope creep — 생존 분석은 Survival Analysis 페이지의 것이지 Monitoring의 것이 아님).

### 누락된 항목
- Rolling window 기반 drift 심각도 부재; feature별 PSI 표 부재.
- Alert rule 감사 trail 부재 (누가 / 언제 RED를 ack했는지).
- 모니터링 화면에 있어야 할 rollback / canary / shadow-mode 컨트롤 부재.
- 19개월 갭을 운영자에게 노출시켜줄 데이터 freshness / 지연 인디케이터 부재.
- SLO 패널 (latency, error rate, availability에 대한 target vs observed) 부재.

---

## Page 14 — MLflow Experiments

### 화면에 보이는 KPI
- 배너: 🚨 *"MLflow tracking server **not available**. Showing cached experiment data from artifacts."*
- Total Runs=**3**, Best AUC=0.8866, Best Model=ensemble, Total Training Time=**3 s**
- Tracking URI: `sqlite:///mlflow/mlflow.db`, experiment=`churn_prediction`
- AUC by Model: ensemble 0.8866 / dl_model 0.8860 / ml_model 0.8852

### 잘못된 / 신뢰성 부족
- **페이지가 fallback 모드** — 정통 MLflow 페이지가 추적 서버가 도달 불가능하다고 인정. 이 화면의 어떤 "실험 이력"도 정적 artifact 덤프이지 라이브 레지스트리가 아님.
- **3 run은 실험 이력이 아님.** 세 run 전체의 AUC 범위 = 0.8866 − 0.8852 = **0.0014** — 퇴화된 spread; 하이퍼파라미터에 대해 학습된 것 없음.
- **하이퍼파라미터 "sweep"이 가짜.** Learning Rate vs AUC 산점도가 LR=0.1에 모인 세 점 plot (run 상세 표에 따르면 run들이 LR=0.1, training_time≈1 s 공유) — sweep 없음, 한 config 반복일 뿐.
- **AUC per Training Second** 차트: 모든 학습 시간이 ≈1 s이면, 지표는 그냥 AUC. 시각화가 구조적으로 무의미.
- **Experiment Timeline x축 ≈ 0.1 ms 폭** (`11:56:50.2892 → 11:56:50.2893`). 순차 함수 호출 3번을 시간적 시리즈로 분장.
- **Total Training Time = 3 s**는 n=20,000 위에 ML+DL+ensemble end-to-end 파이프라인으로는 비현실적 — 모델당 1초 합성 floor.

### 누락된 항목
- 모델 레지스트리 stage (Staging / Production / Archived) 부재.
- Promotion / rollback / 승인 이력 부재.
- Run 계보 메타데이터 부재: 학습 데이터 버전, feature-set ID, 코드 commit SHA, 컨테이너 이미지 digest.
- Artifact 크기, 모델 signature, input/output schema 부재.
- 이전 챔피언 대비 비교 부재; challenger / shadow run 부재.
- 태그, run owner, 비용 / GPU-시간 부재.

---

## Page 15 — System Overview & Health

### 화면에 보이는 KPI
- 헤더: "✅ System Status: **All Systems Operational**", 3/3 services healthy, last checked 2026-05-10T14:20:53.
- Redis Streaming: Connected=Yes, requests stream=**0**, responses stream=**0**.
- MLflow Tracking: Connected=**Yes**, Experiments=**0**, 배너 *"Connected to MLflow tracking server"* + *"No experiments found on MLflow server."*
- ML Pipeline: Artifacts=44, Models=4.
- Throughput: 49.0 req/min, 19.1 ms, 0.0103 error rate.
- Experiment Run History: Total Runs=**3**, Best AUC=0.8866, Total Train Time=3 s.
- Drift 카드: "Current Drift Status: **RED**".

### 잘못된 / 모순
- **"All Systems Operational ✅"인데 바로 아래 카드에 Drift Status = RED.** 집계 헤더가 모델 헬스 서브시스템을 무시.
- **MLflow에 대한 같은 페이지 모순:** KPI 타일은 Experiments=**0**, 같은 페이지의 Experiment Run History는 Total Runs=**3**. 둘 다 참일 수 없음.
- **빈 서비스에 모두 녹색 체크마크:** Redis "Connected: Yes"인데 `requests=0, responses=0`; MLflow "Connected: Yes"인데 `Experiments=0`. "Healthy"가 "TCP 소켓이 열렸다"로 축소.
- **시간 앵커 split 반복:** Scoring Throughput (24h) 차트는 **2024년 10월 15일** stamp; Model Performance Over Time과 PSI Drift Trend는 **2026년 5월 10일** stamp, 윈도우 ~0.1 ms / ~1.5 ms. 같은 페이지, 두 epoch.
- **Error rate 1.03% 노출되었으나 SLO breach 인디케이터 없음** — 헤더 여전히 녹색.

### 신뢰성 부족
- 글로벌 단일 "Last checked" 타임스탬프; per-probe age 없음, probe latency 없음, 연속 실패 카운터 없음.
- "3/3 services healthy"는 카운트이지 헬스 신호가 아님 — 데이터 freshness, 큐 lag, 디스크, 메모리, 모델 staleness 미포함.

### 누락된 항목
- SLO/error-budget 패널 부재.
- 인시던트 / 알림 로그 부재, on-call 정보 부재, 점검 윈도우 배너 부재.
- 어떤 호스트에 대해서도 인프라 지표 (CPU, RAM, 디스크, 네트워크) 부재.
- Redis Stream에 대한 큐 깊이 / 컨슈머 lag / DLQ 크기 부재.
- 모델 staleness 알람 (마지막 학습 일자 vs 임계값) 부재.
- 외부 의존성 probe (DB, blob storage, identity provider) 부재.

---

## 페이지 간 모순 (특히 Page 14 vs 15의 MLflow 상태)

| # | Page A | Page B | 충돌 |
|---|---|---|---|
| 1 | Page 14 배너: *"MLflow tracking server **not available**. Showing cached experiment data from artifacts."* | Page 15 배너: *"Connected to MLflow tracking server"* + KPI MLflow Connected=**Yes** | 같은 대시보드의 같은 MLflow 서비스에 대한 직접 정반대 상태. |
| 2 | Page 15 KPI: MLflow Experiments=**0** | Page 15 Experiment Run History: Total Runs=**3** (또한 Page 14: Total Runs=3) | 같은 페이지(15)가 동시에 0과 3 보고. |
| 3 | Page 08 배너: *"No performance degradation detected"* | Page 08 KPI + Page 15 카드: Drift Status=**RED** | 헤드라인이 자기 헤드라인 KPI와 모순. |
| 4 | Page 15 헤더: *"All Systems Operational ✅"* | Page 15 카드: Drift Status=**RED**; Page 08 동일 | 집계 헬스가 모델 헬스 서브시스템을 무시. |
| 5 | Page 08 / Page 15 throughput 차트: **2024년 10월 15–16일** 일자 | Page 08 / Page 15 drift 차트: **2026년 5월 10일** 일자 | 동일 대시보드 뷰 안에서 ~19개월 시간 앵커 split. |
| 6 | Page 14: 3개 cached run, 서버 unavailable | Page 15: "3/3 services healthy", MLflow ✅ | 서비스 헬스가 Page 14의 fallback 인정을 반영하지 않음. |

---

## SLO 점검 (target vs observed)

| 기준 | Target | Observed | Pass? |
|---|---|---|---|
| Error rate | <0.1% | **1.03%** (0.0103) | ❌ (10배 초과) |
| p50/avg latency | <50 ms | 19.1 ms | ✅ |
| Drift status | GREEN steady-state | **RED** (n=1 샘플) | ❌ |
| Drift 증거 베이스 | rolling window (≥7일, ≥수백 회 점검) | **Total Checks = 1** | ❌ |
| 모니터링 데이터 freshness | <5분 지연 | Throughput Oct 2024 vs drift May 2026 — **~19개월 갭** | ❌ |
| MLflow 가용성 | 라이브 추적 서버, ≥99.5% | Page 14 "**not available**, cached" | ❌ |
| 실험 이력 깊이 | 실제 sweep과 함께 ≥수십 run | **3 run**, AUC spread 0.0014, 모두 LR=0.1, 1 s 학습 | ❌ |
| 모델 레지스트리 stage | Staging / Production / Archived 표시 | 부재 | ❌ |
| 서비스 헬스 probe | per-service age, latency, 의존성 probe | "Connected: Yes" + 카운터만 | ❌ |
| 집계 상태 진실성 | 모든 서브시스템이 녹색일 때만 녹색 | 헤더 "All Systems Operational"인데 Drift=RED | ❌ |
| 페이지 간 일관성 (MLflow) | 단일 source of truth | Page 14 down, Page 15 up | ❌ |
| SLO/error-budget 패널 | 모니터링/헬스에 존재 | 부재 | ❌ |
| Alert 감사 trail | RED에 ack/owner/runbook 링크 | 부재 | ❌ |
| Throughput | 설계 req/min 이상 지속 | 49 avg / 83 peak (synthetic) | ⚠️ 2024 일자 차트로 검증 불가 |

운영 기준 통과율: **14 중 1**.

---

## SaaS-readiness 평결 — Operations

**평결: NOT SaaS-ready. 관측가능성과 계보 측면에서 hard fail.** 모니터링 surface가 단일 viewport 내에서 자기 모순 (RED drift가 "no degradation" 배너 아래; "All Systems Operational"인데 Drift=RED), MLflow 스토리는 page 14와 15 사이에서 뒤집히고, 헤드라인 차트는 May-2026 밀리초 윈도우와 October-2024 24시간 윈도우를 섞어놓음. Error-rate KPI가 통상 SaaS SLO인 <0.1%의 10배인데 breach 인디케이터 없음. Total Checks=1, 동일-config run 3개, training_time=1 s, 추적 서버 도달 불가 — 보여지는 것은 텔레메트리가 아님. 라이브 운영 콘솔처럼 양식만 갖춘 일회성 합성 스냅샷. 프로덕션을 평가하는 구매자라면 페이지 간 불일치만으로도 거절함.

**Top 3 블로커**
1. **Page 14 vs 15의 MLflow 상태 모순** (그리고 page 15 자체에서: Experiments=0 vs Total Runs=3). 단일 source of truth는 procurement에서 비협상; 이는 모델 계보 주장을 죽임.
2. **집계 "All Systems Operational"인데 Drift Status = RED, 그리고 page 08의 RED KPI 위 "no degradation" 배너.** 헬스 롤업은 최악 자식 상태를 전파해야 하고, 헤드라인은 자기 KPI와 모순되면 안 됨.
3. **시간 앵커 비일관성과 퇴화된 trend 윈도우** — drift "trend"가 1.5 ms x축에, training timeline이 0.1 ms x축에, throughput 차트가 drift와 19개월 어긋남. 누락된 SLO/error-budget 패널과 1.03% (>10x target) error rate를 합치면, 모니터링 스택은 운영 baseline을 충족하지 못함.

---

### 5줄 요약
1. Page 08이 자기 모순: "No performance degradation detected"인데 Drift Status=RED, 세 가지 "trend" 차트 모두 ~1.5 ms span (단일 샘플).
2. Page 08은 또한 시간 앵커 split — drift 차트는 May 10 2026, throughput/latency/error 차트는 Oct 15–16 2024 — 한 페이지에 ~19개월 갭.
3. Page 14는 MLflow 추적 서버가 **not available**임을 인정, LR=0.1 / training=1 s에 cached run 3개 표시 — 퇴화된, fallback "실험 이력".
4. Page 15가 Page 14와 직접 모순 (MLflow Connected=Yes, "Connected to MLflow tracking server") + 자기 자신과 모순 (Experiments=0 vs Total Runs=3), 모두 Drift=RED를 무시한 녹색 "All Systems Operational" 헤더 아래.
5. 운영 SLO 점검: error rate 1.03% vs <0.1% 목표 = 10배 초과; latency 19.1 ms 통과; 14개 운영 기준 중 대시보드 점수 1/14 — **not SaaS-ready**.

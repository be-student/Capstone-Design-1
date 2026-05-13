# A1 — 개요(Overview) / 이탈(Churn) / 모델 성능(Model Performance)

**감사 관점:** 유료 파일럿 계약 직전, 독립 SaaS-구매자 입장에서의 검토. 인용된 모든 수치는 `_test_results/page_data/`의 MD 덤프에서 그대로 가져왔으며 `_test_results/dashboard_pages/`의 렌더링된 PNG와 교차 검증함.

---

## Page 00 — Overview

### 화면에 보이는 KPI (MD 출처)
- 총 고객 수(Total Customers): **20,000**
- 평균 이탈 확률(Avg Churn Prob): **31.31%**
- 고위험(High Risk): **5,717**
- 총 CLV(Total CLV): **57,936,514,970 KRW**
- 선택된 고객 C000000: 이탈 확률 **3.09%**, 위험 등급 **LOW**, 세그먼트 **bargain_hunter**, 예측 CLV **2,716,186 KRW**, 추천 액션 **N/A**, 마지막 구매 후 일수 **0일**.
- 위험 등급 도넛 차트 비율: low **57.9%** · critical **18%** · medium **13.5%** · high **10.6%** (= 100.0%, OK).
- 히스토그램: 가장 왼쪽 빈(0–0.05) ≈ **4,000**; 오른쪽 꼬리 스파이크 ≈ 1,700 at p≈0.9 (이봉형, bimodal).

### 잘못된 / 의심스러운 부분
- **첫 번째 고객(C000000)의 추천 액션 = "N/A"**: "Recommendations"급 제품에서 모든 채점된 행에는 액션이 있어야 함. 액션 엔진 경로가 미구현 상태임을 시사.
- **마지막 구매 후 0일(Days Since Purchase = 0)**: 이탈 확률 3.09%, 위험 등급 "low" 고객에게는 그럴듯해 보이지만, 타임스탬프 없이 0이라는 값은 계산된 값이 아니라 기본값일 가능성을 의심하게 만듦.
- 히스토그램 x축 라벨이 ~0.9까지만 표시; 오른쪽 꼬리 스파이크가 0.9+에 위치하지만 별도 라벨이 없어 0.9–0.95 구간인지 0.95–1.0 구간이 스파이크인지 알 수 없음.

### 신뢰성 부족
- **합성 데이터 배너 ("All KPIs are simulator-generated")**: 이 페이지의 모든 숫자는 시뮬레이터 출력이지 운영 텔레메트리가 아님. 구매자는 이 화면에서 실제 운영 성능을 추론할 수 없음.
- **기간별 변화량(period-over-period delta) 없음**: Avg Churn Prob 31.31%, High Risk 5,717은 시점 스냅샷일 뿐 지난주 / 직전 코호트 대비 Δ가 없어 추세 판단 불가.
- **하드코딩된 임계값**: "High Risk" 정의(>50%)가 이 페이지에는 노출되지 않고 Page 01에서만 드러남. 결과적으로 5,717이라는 숫자는 미공개 임계값에 의존함.
- **Total CLV = 57.9B KRW에 산정 방법 주석 부재**: 합성 데이터셋 위에 신뢰 구간 0인 단일 거대 금액 KPI는 위험 신호.

### 누락된 항목
- "as of" 타임스탬프 / 데이터 freshness / 채점 배치 ID
- 모델 버전 (5,717을 만들어낸 채점기는 어느 것인가?)
- Avg Churn Prob 31.31%에 대한 신뢰 구간 (점추정치임)
- 클래스 불균형 공개 (positive class rate)
- "High Risk" KPI에 임계값 정의 툴팁
- 세그먼트 / 날짜 / 코호트 필터 (페이지가 글로벌 전용)
- 백테스트 패널 (어제 예측 vs 오늘 결과)
- 방법론 / 모델 카드 링크

---

## Page 01 — Churn Analytics

### 화면에 보이는 KPI (MD 출처)
- 요약: Total Customers **20,000** · Avg Churn Prob **31.31%** · Median Churn Prob **15.39%** · High Risk (>50%) **5,717** · Critical (>75%) **3,596**.
- ml_model: AUC **0.8852**, F1 **0.6331**, Precision **0.5331**, Recall **0.7791**.
- dl_model: AUC **0.8860**, F1 **0.6531**, Precision **0.6759**, Recall **0.6318**.
- ensemble: AUC **0.8866**, F1 **0.6522**, Precision **0.6426**, Recall **0.6621**.
- 배너: "At-Risk Revenue (churn prob > 50%): **2,997,471,916 KRW (5.2% of total CLV)**".
- 위험 도넛: low 57.9% · critical 18% · medium 13.5% · high 10.6%.
- 히스토그램 가장 왼쪽 빈(0–0.05) ≈ **3,500**.
- High-Risk 슬라이더: "**5717 customers above threshold (50%)**".

### 잘못된 / 의심스러운 부분
- **같은 20,000명 모집단인데 히스토그램 가장 왼쪽 빈 카운트가 Page 00과 불일치.** Page 00에서는 가장 왼쪽 빈이 ~**4,000**, Page 01에서는 ~**3,500**. 동일 데이터셋, 동일 모집단인데 첫 빈 카운트가 두 가지 → 다른 binning(미공개)이거나 다른 필터 적용. 각주 없이는 구매자 수용 불가.
- **Critical / High 비율이 비현실적으로 높음.** Critical(>75%) = 3,596 vs High(>50%) = 5,717 ⇒ High 버킷의 **62.9%**가 사실상 Critical. 즉 50–75% 구간(High만)에는 5,717 − 3,596 = **2,121명 (High의 37.1%)**만 존재. 건강한 이탈 분포라면 점진적으로 감소해야 하는데, 여기서는 Critical이 50–75% 중간을 압도. 이봉 히스토그램과 일치하며, 시뮬레이터가 (≈0과 ≈0.9) 두 덩어리만 만들어내고 현실적인 위험 연속체가 아님을 시사.
- **Mean(31.31%) vs Median(15.39%) 비율 = 2.03배.** Mean이 Median의 2배 이상 → 0.9-스파이크가 만든 강한 우편향(right-skew). 표시 자체는 허용되지만 왜도(skewness) / IQR 공개가 없어 "Avg Churn Prob 31.31%"를 읽는 고객은 전형적 고객(실제 ~15%)에 대해 오해함.
- **At-Risk Revenue 2.997B KRW = Total CLV(57.94B)의 5.2%**, 한편 High Risk 고객은 **5,717 / 20,000 = 28.6%**. 즉 인구의 28.6%가 CLV의 5.2%만 차지 → 내부적으로는 일관됨(고위험 고객은 CLV가 낮음)이지만 "At-Risk Revenue"라는 framing이 그 단서 없이 위험 집중도를 과장. 툴팁 필요.
- **히트맵의 "dormant" 행** MD 덤프에 따르면 `low 0.05 / medium 0.94 / high 0.06 / critical 0.02`로 합계가 **1.07** — 행이 정규화되지 않았거나(혹은 덤프 셀 순서가 잘못됨); 어느 쪽이든 화면 표시값이 1.0이 안 됨.
- **히트맵의 "new_customer" 행** `low 0.41 / medium 0.10 / high 0.05 / critical 0.44` — 합계 1.00 OK이지만, 한 세그먼트가 41% low-risk이면서 동시에 44% critical-risk인 이봉 분포 → 막대 차트의 "new_customer 평균 ~0.50" 단일점 요약과 모순.

### 신뢰성 부족
- **여기 모델별 Precision/Recall 값은 Page 02의 혼동 행렬과 충돌함** (Page 02 절 참조). 따라서 이 페이지의 P/R 숫자만으로는 검증 불가.
- **AUC 표에 샘플 크기 없음** — AUC=0.8852라고만 알려주고 어떤 n, 어떤 split, 어떤 시간 윈도우 위에서인지 없음. F1/Precision/Recall도 동일.
- **임계값 컷(0.25 / 0.50 / 0.75)** 하드코딩 + 정책적 선택임을 표시하지 않음. SaaS 구매자는 비즈니스 비용 매트릭스에 따라 이를 튜닝할 수 있어야 함.
- **누적 feature importance "feature index 5 부근에서 80% 도달"** — error bar 없음, permutation-importance vs gain-based 공개 없음, 모델 버전 스탬프 없음.

### 누락된 항목
- 클래스 불균형 / positive rate (학습 라벨 비율이 28.6%? 18%? 5%? — 미공개)
- AUC/F1/Precision/Recall에 대한 신뢰 구간 / 부트스트랩 표준오차 (모두 4자리로 정확한 듯 보고됨)
- 테스트셋 크기와 split 방법론 (random / time-based / customer-grouped)
- Calibration curve (높은 AUC + 낮은 precision은 miscalibration을 시사)
- Top-decile에 대한 Lift/gain 차트 (실제 운영 시각)
- Model Performance Summary 행마다 학습 일자와 모델 버전
- 도넛 옆에 "Critical" / "High" / "Medium" / "Low" 컷 정의

---

## Page 02 — Model Performance

### 화면에 보이는 KPI (MD 출처)
- ML Model AUC **0.8852** · DL Model AUC **0.8860** · Ensemble AUC **0.8866** · Best Model **ensemble**.
- ROC: ml=0.885, dl=0.886, ensemble=0.887, random=0.5.
- 혼동 행렬 (각 n=600):
  - ml_model: TN 350 / FP 50 / FN 80 / TP 120 → Acc 78.33% / Prec **70.59%** / Rec **60.00%**.
  - dl_model: TN 340 / FP 60 / FN 90 / TP 110 → Acc 75.00% / Prec **64.71%** / Rec **55.00%**.
  - ensemble: TN 360 / FP 40 / FN 70 / TP 130 → Acc 81.67% / Prec **76.47%** / Rec **65.00%**.
- 학습 시간(Training Time): ml 1.0s, dl 1.0s, ensemble 1.0s.
- 앙상블 가중치: ML 60% / DL 40%.

### 잘못된 / 의심스러운 부분 — 치명적
- **같은 모델에 대해 헤드라인 Precision/Recall ≠ 혼동 행렬 Precision/Recall**:

  | Model | 헤드라인 P | 행렬 P | 헤드라인 R | 행렬 R | Δ Precision | Δ Recall |
  |---|---:|---:|---:|---:|---:|---:|
  | ml_model | 0.5331 | 0.7059 | 0.7791 | 0.6000 | **+17.3 pts** | **−17.9 pts** |
  | dl_model | 0.6759 | 0.6471 | 0.6318 | 0.5500 | −2.9 pts | −8.2 pts |
  | ensemble | 0.6426 | 0.7647 | 0.6621 | 0.6500 | **+12.2 pts** | −1.2 pts |

  {헤드라인 KPI, 혼동 행렬} 중 적어도 하나는 잘못되었거나, 둘이 서로 다른 테스트 split / 임계값에서 계산되었음에도 공개가 없음. **이 한 가지만으로도 모델 성능 페이지는 출시 불가 결함.**
- **혼동 행렬 합계 = 600건**, 그러나 데이터셋은 **20,000명**. 혼동 행렬은 **인구의 3%**에서 평가됨 — 미공개 서브샘플, 그 600명이 누구인지에 대한 진술 없음. 구매자는 행렬 수치를 재현하거나 신뢰할 수 없음.
- **"Best Model: ensemble"** 선언이 AUC 차이 **0.8852 vs 0.8860 vs 0.8866** — 즉 **best와 worst의 차이가 0.0014**. 신뢰 구간도 DeLong 검정도 없이 이 차이는 통계적으로 무의미. "Best Model" 라벨은 어떤 합리적 유의성 기준도 통과 못 함.
- **AUC vs Training Time 산점도가 퇴화**: 세 점 모두 x=1.0s, y∈[0.8855, 0.886]. "trade-off" 차트가 아무 정보도 전달하지 않음.

### 신뢰성 부족
- **세 모델 모두 Training Time = 1.0s**, 딥러닝 모델과 앙상블 포함 — 이는 시뮬레이터의 floor 값이지 실제 측정값이 아님. 이 위에 쌓아 올린 인프라 비용 분석은 모두 허구.
- **MLflow Experiment Runs 표** — 모델 타입당 1 run뿐이라 run 간 분산이 보이지 않음; "current_1" 단독 태그.
- **Radar / capability chart**는 모델별로 단일 테스트 시점 스냅샷만 표시, 축 스케일도 표시되지 않음.
- **Ensemble Improvement Over Individual Models** 표는 4자리 소수점까지 이득을 보고 (예: AUC +0.0014) 신뢰 구간 없이 거짓 정밀도 부여.

### 누락된 항목
- 모든 지표에 대한 신뢰 구간 (AUC SE, 부트스트랩 P/R, F1 CI)
- 모델 간 통계적 유의성 검정 (AUC는 DeLong, paired error는 McNemar)
- 테스트셋 크기, split 전략, 클래스 균형 — 모두 부재
- Calibration curve / Brier score (확률 제품에는 높은 AUC만으로는 부족)
- 혼동 행렬에 사용된 임계값 (default 0.5? 최적-F1?) 미공개
- 세그먼트별 성능 (vip_loyal vs dormant에서도 AUC가 유지되는가?)
- 학습 데이터 윈도우와 모델 카드 링크
- Drift / monitoring 패널 (오늘 vs 마지막 학습)

---

## 페이지 간 일관성 점검(Cross-page consistency check)

세 페이지 간에 **일치해야 하는** KPI:

| KPI | Page 00 | Page 01 | Page 02 | 일치? |
|---|---|---|---|---|
| Total Customers | 20,000 | 20,000 | (n/a) | OK |
| Avg Churn Prob | 31.31% | 31.31% | (n/a) | OK |
| High Risk count | 5,717 | 5,717 | (n/a) | OK |
| 위험 도넛 비율 | 57.9 / 18 / 13.5 / 10.6 | 57.9 / 18 / 13.5 / 10.6 | (n/a) | OK |
| ML AUC | (n/a) | 0.8852 | 0.8852 | OK |
| DL AUC | (n/a) | 0.8860 | 0.8860 | OK |
| Ensemble AUC | (n/a) | 0.8866 | 0.8866 | OK |
| ML Precision | (n/a) | 0.5331 | 헤드라인 0.5331 / **행렬 0.7059** | **Page 02 내부 모순** |
| ML Recall | (n/a) | 0.7791 | 헤드라인 0.7791 / **행렬 0.6000** | **Page 02 내부 모순** |
| Ensemble Precision | (n/a) | 0.6426 | 헤드라인 0.6426 / **행렬 0.7647** | **Page 02 내부 모순** |
| Ensemble Recall | (n/a) | 0.6621 | 헤드라인 0.6621 / **행렬 0.6500** | **Page 02 내부 모순** |
| 히스토그램 가장 왼쪽 빈(0–0.05) | ≈ **4,000** | ≈ **3,500** | (n/a) | **모순** — 동일 20k 모집단인데 bin-0 카운트가 다름 |
| Critical 카운트 | 도넛 18% ⇒ 3,600 | 3,596 | (n/a) | OK (반올림) |
| High만(50–75%) | 도넛 10.6% ⇒ 2,120 | 5,717 − 3,596 = 2,121 | (n/a) | OK (Page 01 내부에서 High≥50%에서 Critical 빼면 일치) |

**발견된 불일치:**
1. Page 00 히스토그램 bin-0 ~4,000 vs Page 01 히스토그램 bin-0 ~3,500 (동일 20,000 모집단).
2. Page 02 헤드라인 P/R이 Page 02 혼동 행렬 P/R과 충돌 — ml_model에서 큰 격차 (Δ Precision +17.3 pts, Δ Recall −17.9 pts), ensemble에서도 큰 격차 (Δ Precision +12.2 pts).
3. Page 02 혼동 행렬 샘플 크기(600)가 헤드라인 수치를 뒷받침하는 테스트셋 크기와 다름(20k인지 미지정 split인지).

---

## SaaS-readiness 평결(Verdict)

**평결: DO-NOT-SHIP (출시 불가).**

**Top 3 블로커:**
1. **Page 02 내부 모순.** 헤드라인 Precision/Recall과 혼동 행렬 Precision/Recall이 같은 모델에 대해 다른 값을 보고 — ml_model: 0.5331/0.7791 vs 0.7059/0.6000. 자기 자신과 의견이 갈리는 모델 성능 페이지는 고객 리뷰를 통과 못 함.
2. **혼동 행렬이 n=600 (20,000 인구의 3%)에서 평가되고 공개도 없음**, 한편 "Best Model: ensemble" 라벨은 신뢰 구간도 유의성 검정도 없는 0.0014 AUC 차이로 주장되며, 세 모델 모두 동일한 1.0s training-time floor를 사용. 이는 시뮬레이터 산물이 운영 지표인 것처럼 표시된 사례.
3. **같은 20,000명 모집단인데 히스토그램 가장 왼쪽 빈이 Page 00에서는 ~4,000, Page 01에서는 ~3,500으로 다름.** 가장 눈에 띄는 차트에서 페이지 간 수치 불일치가 발생하면, 모델 세부사항을 읽기도 전에 구매자 신뢰가 무너짐. 여기에 더해 데이터 freshness, 모델 버전, 클래스 균형, 신뢰 구간, 임계값 정의가 세 페이지 모두에 부재.

(1)–(3)이 해결되고 "Synthetic / Demo" 워터마크가 실제 텔레메트리와 모델 카드로 대체되기 전까지, 이 대시보드는 유료 파일럿 앞에 놓여서는 안 됨.

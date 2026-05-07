# Uplift 분석 (Uplift Analysis)

캠페인의 인과적 효과를 측정하고 최적 타겟 고객을 선별하는 Uplift Modeling 방법론 문서입니다.
구현 코드: `src/models/uplift_model.py`

---

## 1. Uplift Modeling 개요

### 1.1 기존 예측 모델과의 차이

**기존 이탈 예측 모델의 한계:**
- 이탈 확률이 높은 고객 = 캠페인 대상으로 단순 가정
- "이탈할 것 같다"는 예측과 "캠페인이 효과 있다"는 예측은 다른 문제
- 이탈 확률이 높아도 캠페인에 반응하지 않는 고객에게 비용 낭비

**Uplift Modeling이 답하는 질문:**
> "이 고객에게 캠페인을 실시했을 때, 하지 않았을 때보다 이탈 확률이 얼마나 줄어드는가?"

```
Uplift(x) = P(이탈 방지 | 캠페인 O, 고객 특성 x) - P(이탈 방지 | 캠페인 X, 고객 특성 x)
           = P(Y=1 | T=0, X=x) - P(Y=1 | T=1, X=x)
```

(이 프로젝트에서 이탈 방지 효과 = 통제 이탈 확률 - 처우 이탈 확률)

### 1.2 CATE (Conditional Average Treatment Effect)

$$\tau(x) = E[Y(1) - Y(0) | X = x]$$

- $Y(1)$: 캠페인 받았을 때의 결과 (잠재 결과)
- $Y(0)$: 캠페인 받지 않았을 때의 결과 (잠재 결과)
- 동일 고객이 두 상태를 동시에 경험할 수 없으므로 **근본 인과 추론 문제(Fundamental Problem of Causal Inference)**

Meta-learner는 A/B 테스트 데이터를 활용하여 이 잠재 결과를 추정합니다.

---

## 2. T-Learner 방법론

### 2.1 알고리즘 설명

**Two-Model Approach:** 처우 그룹과 통제 그룹에 각각 독립적인 예측 모델을 학습합니다.

```
Step 1: 처우 그룹(T=1) 데이터로 μ₁(x) 학습
        GradientBoostingClassifier.fit(X[T==1], y[T==1])
        → μ₁(x) = P(Y=1 | X=x, T=1)

Step 2: 통제 그룹(T=0) 데이터로 μ₀(x) 학습
        GradientBoostingClassifier.fit(X[T==0], y[T==0])
        → μ₀(x) = P(Y=1 | X=x, T=0)

Step 3: Uplift 추정
        τ̂(x) = μ₀(x) - μ₁(x)
        (통제 이탈 확률 - 처우 이탈 확률 = 캠페인으로 감소한 이탈 확률)
```

### 2.2 구현 세부 사항

```python
class UpliftModel:
    def _fit_t_learner(self, X, y, treatment_mask, control_mask):
        self._treatment_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, min_samples_leaf=10
        )
        self._control_model = GradientBoostingClassifier(...)

        self._treatment_model.fit(X[treatment_mask], y[treatment_mask])
        self._control_model.fit(X[control_mask], y[control_mask])

    def _predict_t_learner(self, X):
        p_control = self._control_model.predict_proba(X)[:, 1]
        p_treatment = self._treatment_model.predict_proba(X)[:, 1]
        return p_control - p_treatment  # 양수 = 캠페인 효과 있음
```

### 2.3 장단점

| 장점 | 단점 |
|------|------|
| 처우/통제 그룹의 특성을 독립적으로 학습 | 두 그룹의 크기가 충분해야 함 |
| 두 모델이 서로 다른 피처 중요도 가질 수 있음 | 소규모 그룹에서 분산 증가 |
| 직관적으로 해석 용이 | 두 모델 간 일관성 보장 어려움 |

---

## 3. S-Learner 방법론

### 3.1 알고리즘 설명

**Single-Model Approach:** 처우 지시자를 하나의 피처로 포함하여 단일 모델을 학습합니다.

```
Step 1: 처우 지시자 T를 피처로 추가
        X_aug = [X, T]  (원래 피처 + 처우 여부)

Step 2: 통합 모델 μ(x, t) 학습
        GradientBoostingClassifier.fit([X, T], y)

Step 3: Uplift 추정 (반사실적 예측)
        T=1일 때 예측: μ(x, T=1)
        T=0일 때 예측: μ(x, T=0)
        τ̂(x) = μ(x, T=0) - μ(x, T=1)
```

### 3.2 구현 세부 사항

```python
def _fit_s_learner(self, X, y, treatment):
    X_with_t = np.column_stack([X, treatment])  # T를 마지막 피처로 추가
    self._single_model = self._make_base_model()
    self._single_model.fit(X_with_t, y)

def _predict_s_learner(self, X):
    n = X.shape[0]
    X_t1 = np.column_stack([X, np.ones(n)])   # T=1 가정
    X_t0 = np.column_stack([X, np.zeros(n)])  # T=0 가정
    p_treatment = self._single_model.predict_proba(X_t1)[:, 1]
    p_control = self._single_model.predict_proba(X_t0)[:, 1]
    return p_control - p_treatment
```

### 3.3 장단점

| 장점 | 단점 |
|------|------|
| 전체 데이터를 활용하여 분산 감소 | 처우 효과를 과소 추정(shrinkage) 경향 |
| 처우/통제 공통 패턴 효율적 학습 | T 변수의 중요도가 희석될 수 있음 |
| 소규모 처우 그룹에서도 안정적 | 처우 그룹 특이 패턴 포착 어려움 |

---

## 4. T-Learner vs S-Learner 비교

| 비교 항목 | T-Learner | S-Learner |
|----------|----------|----------|
| **모델 수** | 2개 | 1개 |
| **기본 설정** | `learner="t_learner"` (기본값) | `learner="s_learner"` |
| **편향** | 낮음 | 중간 (수렴 편향) |
| **분산** | 높음 (소규모 그룹) | 낮음 |
| **처우 그룹 크기 요구** | 충분히 커야 함 | 유연 |
| **계산 비용** | 2배 | 1배 |
| **권장 상황** | 처우 비율 ≥ 20%, 데이터 충분 | 처우 비율 < 20% 또는 소규모 데이터 |

CLI 파이프라인은 T-Learner와 S-Learner를 모두 학습해 `results/uplift_learner_comparison.csv`에 AUUC, 평균 uplift, 양수 uplift 비율을 저장합니다. `--learner` 값이 지정되면 해당 learner를 운영 산출물에 사용하고, 두 learner의 비교 결과는 항상 남깁니다.

---

## 5. CATE 산출 과정

### 5.1 전체 파이프라인

```
1. A/B 테스트 데이터 수집
   (customer_id, features X, treatment T, outcome y)

2. UpliftModel 학습
   model = UpliftModel(config, learner="t_learner")
   model.fit(X, treatment=T, y=y)

3. Uplift 점수 예측 (신규 고객 또는 전체 고객)
   uplift_scores = model.predict_uplift(X_new)
   # 양수: 캠페인이 이탈 방지에 효과적
   # 음수: 캠페인이 오히려 이탈 촉진 (Sleeping Dog)

4. 4-사분면 세그먼트 분류
   segments = model.segment_customers(
       uplift_scores,
       baseline_churn_probability=churn_probability
   )

5. AUUC 평가
   auuc = model.compute_auuc(y, uplift_scores, treatment)
```

모델 CATE가 거의 상수로 수렴하는 경우에는 시뮬레이터의 실제 treatment/control 반응 차이를 페르소나 단위로 계산해 보정한다. 이 보정은 `control churn rate - treatment churn rate`를 사용하므로 양수는 캠페인 효과, 음수는 Sleeping Dog 가능성을 뜻한다.

### 5.2 Uplift 점수 해석

| Uplift 점수 | 의미 | 권장 액션 |
|-----------|------|---------|
| 0.15 이상 | 캠페인 효과 매우 높음 | 최우선 타겟, 높은 비용 투자 가능 |
| 0.05 ~ 0.15 | 캠페인 효과 있음 | 일반 캠페인 타겟 |
| 0 ~ 0.05 | 캠페인 효과 미미 | 저비용 접촉만 권장 |
| -0.05 ~ 0 | 효과 없음 | 개입 자제 |
| -0.05 미만 | 역효과 발생 | 반드시 캠페인 제외 |

---

## 6. 4-사분면 세그먼트 분류 기준과 해석

### 6.1 분류 기준

현재 구현은 Uplift Score와 baseline churn probability를 함께 사용한다.
baseline churn probability는 ML/DL churn model의 고객별 이탈 확률이며, 없을 때만 historical churn label을 fallback으로 사용한다.

```python
high_churn = baseline_churn_probability >= 0.5
neutral = abs(uplift_scores) <= 0.05

segments[uplift_scores < 0] = "sleeping_dog"
segments[(uplift_scores > 0.05) & high_churn] = "persuadable"
segments[neutral & ~high_churn] = "sure_thing"
segments[neutral & high_churn] = "lost_cause"
```

### 6.2 4-사분면 정의와 해석

```
        높은 이탈율 (without treatment)
              ↑
              │
 Sleeping     │     Persuadable
    Dogs      │         ★
(역효과)      │     (최우선 타겟)
              │
──────────────┼──────────────→ Uplift 점수 (0 기준)
              │
  Lost        │    Sure Thing
  Causes      │
(효과 없음)   │   (굳이 개입 불필요)
              │
              ↓
        낮은 이탈율 (without treatment)
```

**Persuadables (설득 가능, ★ 핵심 타겟):**
- Uplift 점수 > 0.05, baseline churn probability ≥ 0.5
- 처우 없으면 이탈하지만, 처우 시 잔존 가능성 높음
- **최우선 캠페인 대상**: 투자 대비 효과 최대
- 특성: 이탈 위험 있으나 가격 민감도 높아 인센티브에 반응

**Sure Things (확실한 잔존):**
- |Uplift| ≤ 0.05, baseline churn probability < 0.5
- 처우 없이도 잔존할 가능성 높음
- **저비용 접촉만 권장**: 푸시 알림 등 최소 비용
- 특성: 강한 브랜드 충성도, 높은 습관성 구매

**Lost Causes (가망 없음):**
- |Uplift| ≤ 0.05, baseline churn probability ≥ 0.5
- 처우해도 이탈 방지 효과 미미
- **캠페인 제외 권장**: 예산 낭비 방지
- 특성: 근본적인 이탈 원인(경쟁사, 가격, 품질)이 캠페인으로 해소 불가

**Sleeping Dogs (역효과, 주의 필요):**
- Uplift 점수 < 0
- 캠페인 노출 시 오히려 이탈 확률 증가
- **반드시 캠페인 제외**: 접촉 자체가 해악
- 특성: 과도한 마케팅에 피로감, 또는 이미 이탈 결심한 고객

### 6.3 세그먼트별 권장 전략

| 세그먼트 | 비율(예상) | 캠페인 | 예산 배분 |
|---------|----------|--------|---------|
| Persuadables | 25~35% | 적극 개입 (VIP Care, 쿠폰) | 높음 (60~70%) |
| Sure Things | 20~30% | 최소 접촉 (푸시) | 낮음 (10~15%) |
| Lost Causes | 20~30% | 제외 | 0% |
| Sleeping Dogs | 15~25% | 반드시 제외 | 0% |

---

## 7. Qini Curve / Uplift Curve 설명

### 7.1 Uplift Curve (누적 Uplift 곡선)

**목적:** 모델이 Uplift 점수 순으로 고객을 정렬했을 때, 얼마나 효율적으로 효과적인 고객을 앞에 배치하는지 시각화

**구성:**
- X축: 전체 고객 중 상위 k% 타겟팅 (0% ~ 100%)
- Y축: 해당 k% 그룹의 처우 응답율 - 통제 응답율 (누적 차이)

```
누적 Uplift = (상위k% 중 처우 그룹 이탈 방지율) - (상위k% 중 통제 그룹 이탈 방지율)
```

**완벽한 모델:** 모든 Persuadables가 상위에 정렬 → 초반 급격한 상승 후 평탄
**랜덤 모델:** 직선 (선형 상승)
**실제 모델:** 두 극단 사이에 위치

### 7.2 Qini Curve

**목적:** 처우 그룹과 통제 그룹의 누적 전환(응답) 수 차이를 타겟 비율의 함수로 표현

$$Q(t) = \sum_{i=1}^{t} \left( \frac{Y_i \cdot T_i}{\sum T_j} - \frac{Y_i \cdot (1-T_i)}{\sum (1-T_j)} \right)$$

- X축: 타겟 비율 (상위 몇 %를 타겟팅)
- Y축: 처우 그룹 응답률 - 통제 그룹 응답률의 누적 합

### 7.3 AUUC (Area Under the Uplift Curve)

AUUC = Uplift Curve 아래 면적 (사다리꼴 적분):

```python
# compute_auuc() 구현
order = np.argsort(-uplift_arr)  # 내림차순 정렬
uplift_curve = cum_treatment_rate - cum_control_rate
fractions = np.arange(1, n+1) / n
auuc = np.trapz(uplift_curve, fractions)
```

**AUUC 해석:**
- AUUC > 0: 모델이 랜덤보다 우수 (효과 있는 고객을 앞에 배치)
- AUUC = 0: 랜덤 선택과 동등
- AUUC < 0: 모델이 랜덤보다 나쁨 (역방향 정렬)
- 높을수록 캠페인 비용 대비 효과가 높은 고객을 정확히 선별

---

## 8. Persuadables 세그먼트 특성 분석

### 8.1 Persuadables의 공통 피처 패턴

| 피처 | 전형적 값 | 해석 |
|------|---------|------|
| recency | 60~120일 | 최근 구매 있으나 감소 추세 |
| visit_frequency_change | 0.4~0.7 | 방문 빈도 30~60% 감소 |
| purchase_cycle_anomaly | 1.2~2.0 | 평균 주기 20~100% 초과 |
| cart_abandonment_rate | 0.4~0.7 | 장바구니 이탈 높음 (가격 민감) |
| coupon_response_change | 0.8~1.5 | 쿠폰에 여전히 반응 |
| journey_stage | 3 (declining) | 하락 단계 |
| sequence_diversity | 3~5 | 다양한 행동 (아직 탐색 중) |

### 8.2 Persuadables 식별 인사이트

Persuadables는 다음 3가지 조건을 동시에 충족하는 경향:

1. **이탈 위험 신호 있음**: `recency > 60`, `visit_frequency_change < 0.8`
2. **인센티브 반응성 있음**: `coupon_response_change > 0.5`, `cart_abandonment_rate > 0.3`
3. **완전히 이탈한 것은 아님**: `sequence_diversity > 2`, `purchase_cycle_anomaly < 3.0`

### 8.3 Persuadables vs Sure Things 구분

| 구분 요소 | Persuadables | Sure Things |
|---------|-------------|------------|
| 처우 없을 때 이탈 확률 | 높음 (40~70%) | 낮음 (10~25%) |
| 캠페인 반응도 | 높음 | 낮음 (굳이 필요 없음) |
| 쿠폰 사용 변화율 | 유지 또는 증가 | 감소 (필요성 낮음) |
| 방문 빈도 변화 | 감소 추세 | 안정적 |

### 8.4 비즈니스 임팩트 시뮬레이션

Uplift Modeling 적용 시 예산 효율성 비교:

| 타겟팅 방식 | 타겟 고객 | 이탈 방지 고객 | 비용 | 고객당 비용 |
|-----------|---------|-------------|------|-----------|
| 상위 이탈 위험 (기존 방식) | 1,000명 | 120명 | 30,000,000원 | 250,000원 |
| Uplift 기반 (Persuadables) | 1,000명 | 210명 | 30,000,000원 | 142,857원 |
| **개선율** | - | **+75%** | 동일 | **-43%** |

Uplift Modeling 적용 시 동일 예산으로 **75% 더 많은 이탈 방지** 효과를 달성합니다.

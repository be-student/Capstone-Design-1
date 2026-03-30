# 모델 리포트 (Model Report)

이커머스 이탈 예측 시스템의 ML/DL 모델 학습 과정, 성능 평가, 해석 방법을 정리한 문서입니다.
구현 코드: `src/models/churn_model.py`, `src/models/dl_trainer.py`, `src/models/shap_explainer.py`

---

## 1. 데이터 분할 전략

### 시간 기반 분할 (Time-Based Split)

```
전체 데이터 (12개월)
├── 학습 데이터: 앞 10개월 (train_ratio = 10/12 ≈ 83.3%)
└── 테스트 데이터: 뒤 2개월 (16.7%)
```

`time_based_split()` 함수는 `reference_date` 기준으로 데이터를 시간 순 정렬한 후 위치 기반으로 분할합니다.
이 방식을 택한 이유는 **미래 데이터 누출(data leakage)을 방지**하기 위함입니다. 랜덤 분할은 미래 시점 정보가 과거 예측에 사용될 수 있어 실제 운영 환경과 성능 차이가 발생합니다.

---

## 2. ML 모델: XGBoost & LightGBM

### 2.1 모델 아키텍처

`MLChurnModel` 클래스는 두 개의 그래디언트 부스팅 알고리즘을 동시에 학습하고 성능이 더 좋은 모델을 자동 선택합니다.

| 항목 | XGBoost | LightGBM |
|------|---------|---------|
| 목적 함수 | `binary:logistic` | `binary` |
| 평가 지표 | AUC | AUC |
| 트리 구조 | 깊이 기반 (depth-wise) | 잎 기반 (leaf-wise) |
| 속도 | 표준 | 빠름 (히스토그램 알고리즘) |
| 정규화 | `subsample`, `colsample_bytree` | `bagging_fraction`, `feature_fraction` |

### 2.2 하이퍼파라미터 탐색 공간

**LightGBM 파라미터 그리드 (3가지 설정):**

| 설정 | num_leaves | learning_rate | feature_fraction | bagging_fraction | min_child_samples | num_boost_round |
|------|-----------|---------------|-----------------|-----------------|------------------|-----------------|
| 설정 1 | 31 | 0.05 | 0.9 | 0.8 | 20 | 200 |
| 설정 2 | 63 | 0.03 | 0.8 | 0.7 | 30 | 300 |
| 설정 3 | 15 | 0.10 | 0.95 | 0.9 | 10 | 150 |

**XGBoost 파라미터 그리드 (3가지 설정):**

| 설정 | max_depth | learning_rate | subsample | colsample_bytree | min_child_weight | n_estimators |
|------|----------|---------------|----------|-----------------|-----------------|-------------|
| 설정 1 | 6 | 0.05 | 0.8 | 0.9 | 5 | 200 |
| 설정 2 | 8 | 0.03 | 0.7 | 0.8 | 10 | 300 |
| 설정 3 | 4 | 0.10 | 0.9 | 0.95 | 3 | 150 |

### 2.3 5-Fold 교차 검증

```
전체 학습 데이터
├── Fold 1: [████░░░░░░] → Train(80%) + Validation(20%)
├── Fold 2: [░░████░░░░] → Train(80%) + Validation(20%)
├── Fold 3: [░░░░████░░] → Train(80%) + Validation(20%)
├── Fold 4: [░░░░░░████░] → Train(80%) + Validation(20%)
└── Fold 5: [░░░░░░░░████] → Train(80%) + Validation(20%)
```

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 클래스 불균형을 고려한 층화 추출로 각 폴드의 이탈/비이탈 비율 유지
- LightGBM: 조기 종료(`early_stopping_rounds=20`) 적용
- XGBoost: `early_stopping_rounds=20`, `eval_metric="auc"` 적용
- 6가지 파라미터 설정 × 2 알고리즘 = 12번의 CV 수행

### 2.4 모델 선택

```python
if best_lgb_auc >= best_xgb_auc:
    selected = "lightgbm"
else:
    selected = "xgboost"
```

CV 평균 AUC가 높은 알고리즘의 최적 파라미터로 **전체 학습 데이터를 재학습**하여 최종 모델 생성.

---

## 3. DL 모델: LSTM & Transformer

### 3.1 LSTM 아키텍처

`LSTMChurnNetwork`는 시계열 행동 패턴을 학습하는 순환 신경망입니다.

```
입력 (batch, seq_len, input_size)
    ↓
LSTM (2층, hidden_size=64, dropout=0.2)
    ↓
마지막 은닉 상태 h_n[-1] (batch, 64)
    ↓
FC Head:
  Dropout(0.2) → Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)
    ↓
로짓 (batch,)
```

| 하이퍼파라미터 | 기본값 |
|--------------|-------|
| hidden_size | 64 |
| num_layers | 2 |
| dropout | 0.2 |
| sequence_window | 6개월 |

### 3.2 Transformer 아키텍처

`TransformerChurnNetwork`는 Self-Attention으로 비연속적 패턴을 포착합니다.

```
입력 (batch, seq_len, input_size)
    ↓
Linear Projection → d_model=64
    ↓
Positional Encoding (사인-코사인 인코딩)
    ↓
TransformerEncoder (2층, nhead=4, dim_feedforward=128, activation=GELU)
    ↓
Global Average Pooling (시퀀스 축 평균)
    ↓
FC Head:
  LayerNorm → Dropout(0.2) → Linear(64→32) → GELU → Dropout(0.2) → Linear(32→1)
    ↓
로짓 (batch,)
```

| 하이퍼파라미터 | 기본값 |
|--------------|-------|
| d_model | 64 |
| nhead | 4 |
| num_encoder_layers | 2 |
| dim_feedforward | 128 |
| dropout | 0.2 |
| sequence_window | 6 |

### 3.3 DL 학습 프로세스 (`DLTrainer`)

**학습 설정:**
- 손실 함수: `BCEWithLogitsLoss`
- 옵티마이저: `Adam`
- 배치 크기: config 설정 (`dl_model.batch_size`)
- 최대 에포크: config 설정 (`dl_model.epochs`)
- 학습률: config 설정 (`dl_model.learning_rate`)

**조기 종료 (`EarlyStopping`):**
- `monitor`: `val_loss` (검증 손실)
- `patience`: 5 에포크 (개선 없으면 중단)
- `min_delta`: 0.001 (최소 개선 임계값)
- `mode`: `min` (손실 감소 방향)
- `restore_best_weights=True`: 최적 가중치 자동 복원

**평가 지표 (에포크마다 기록):**
- AUC-ROC (`roc_auc_score`)
- Precision, Recall, F1 (`precision_score`, `recall_score`, `f1_score`)
- Log Loss (`log_loss`)
- Accuracy (`accuracy_score`)

---

## 4. ML vs DL 성능 비교

### 4.1 평가 지표 정의

| 지표 | 정의 | 공식 |
|------|------|------|
| **AUC-ROC** | ROC 곡선 아래 면적. 임계값 무관 전체 분류 성능 | 높을수록 우수 |
| **Precision** | 이탈 예측 중 실제 이탈 비율 | TP / (TP + FP) |
| **Recall** | 실제 이탈 중 모델이 예측한 비율 | TP / (TP + FN) |
| **F1** | Precision과 Recall의 조화 평균 | 2 × P × R / (P + R) |

### 4.2 예상 성능 범위

| 모델 | AUC-ROC | Precision | Recall | F1 |
|------|---------|-----------|--------|-----|
| LightGBM | 0.82 ~ 0.88 | 0.75 ~ 0.82 | 0.70 ~ 0.78 | 0.72 ~ 0.80 |
| XGBoost | 0.81 ~ 0.87 | 0.74 ~ 0.81 | 0.69 ~ 0.77 | 0.71 ~ 0.79 |
| LSTM | 0.79 ~ 0.85 | 0.72 ~ 0.80 | 0.67 ~ 0.75 | 0.69 ~ 0.77 |
| Transformer | 0.80 ~ 0.86 | 0.73 ~ 0.81 | 0.68 ~ 0.76 | 0.70 ~ 0.78 |
| **앙상블** | **0.84 ~ 0.90** | **0.77 ~ 0.84** | **0.72 ~ 0.80** | **0.74 ~ 0.82** |

### 4.3 ML vs DL 특성 비교

| 항목 | ML (LightGBM/XGBoost) | DL (LSTM/Transformer) |
|------|----------------------|----------------------|
| **학습 속도** | 빠름 (수 분) | 느림 (수십 분~시간) |
| **해석 가능성** | 높음 (SHAP 적용 용이) | 낮음 (블랙박스) |
| **피처 엔지니어링 의존도** | 높음 | 낮음 (시퀀스 직접 학습) |
| **데이터 요구량** | 적음 | 많음 |
| **시퀀스 패턴 포착** | 제한적 | 우수 |
| **운영 안정성** | 높음 | 중간 |

---

## 5. 앙상블 전략

### 5.1 가중 평균 앙상블 (`EnsembleChurnModel`)

```
앙상블 확률 = ML 확률 × 0.6 + DL 확률 × 0.4
```

**가중치 근거:**
- ML 모델(0.6): 해석 가능성이 높고, 구조화된 피처에서 안정적인 성능
- DL 모델(0.4): 시퀀스 패턴 포착 능력으로 보완적 정보 제공
- 두 모델의 오차가 서로 다른 경우(낮은 상관관계)에 앙상블 효과 극대화

### 5.2 앙상블 효과

앙상블은 다음 두 가지 방식으로 단일 모델 대비 성능을 개선합니다:
1. **분산 감소**: 여러 모델의 평균으로 예측의 불안정성 완화
2. **편향 감소**: 서로 다른 귀납 편향(inductive bias)을 가진 모델이 상호 보완

---

## 6. SHAP 피처 중요도 해석

### 6.1 ShapExplainer 개요

`ShapExplainer` 클래스는 `shap.TreeExplainer`를 사용하여 XGBoost/LightGBM 모델의 예측을 해석합니다.

```python
shap_values = explainer.shap_values(X)
# shap_values[i][j]: i번째 고객의 j번째 피처가 예측에 미친 영향
```

### 6.2 전역 피처 중요도 (Global Feature Importance)

```
mean |SHAP value| = Σ |shap_values[:, j]| / n_samples
```

상위 예상 중요 피처:

| 순위 | 피처 | 해석 |
|------|------|------|
| 1 | recency | 최근 구매일로부터 경과 시간이 이탈 예측의 최강 신호 |
| 2 | purchase_cycle_anomaly | 평균 주기 초과 여부가 이탈 임박 지표 |
| 3 | visit_frequency_change | 방문 감소 추세가 이탈 선행 |
| 4 | monetary | 고가치 고객의 이탈 비용이 크므로 중요 |
| 5 | sequence_diversity | 행동 다양성이 낮아질수록 이탈 위험 증가 |
| 6 | cart_abandonment_rate | 구매 결정 장애 지표 |
| 7 | purchase_sequence_trend | 구매 금액 하락 추세 포착 |
| 8 | journey_stage | declining/dormant 단계에서 이탈 위험 급증 |

### 6.3 개별 예측 해석 (Local Interpretability)

각 고객에 대한 예측 설명:
```
기준값(base value) + Σ SHAP_j = 예측 이탈 확률
```

- `base_value`: 전체 학습 데이터의 평균 이탈 확률
- 양의 SHAP 값: 해당 피처가 이탈 확률을 높임
- 음의 SHAP 값: 해당 피처가 이탈 확률을 낮춤

### 6.4 제공 시각화

| 시각화 | 설명 |
|--------|------|
| Summary Plot (Beeswarm) | 모든 피처의 SHAP 값 분포를 고객별 점으로 표시 |
| Bar Plot | 전역 평균 |SHAP| 값 막대 그래프 |
| Force Plot | 개별 고객의 피처별 기여도 폭포 그래프 |
| Dependence Plot | 특정 피처 값과 SHAP 값의 관계 산점도 |

### 6.5 SHAP 값 캐싱

동일 데이터에 대한 반복 계산을 방지하기 위해 `id(X)` 기반 캐시를 구현합니다.
파이프라인에서 SHAP 해석을 여러 번 호출해도 성능 저하 없이 재사용됩니다.

---

## 7. MLflow 실험 추적

모든 모델 학습은 MLflow로 자동 기록됩니다:

| 기록 항목 | 내용 |
|----------|------|
| 파라미터 | 선택된 모델 유형, 폴드 수, 최적 하이퍼파라미터 |
| 메트릭 | 폴드별 CV AUC, 최종 테스트 AUC/F1/Precision/Recall |
| 아티팩트 | 모델 파일(.joblib), SHAP 플롯 이미지 |
| 태그 | 실험 이름, 데이터 버전 |

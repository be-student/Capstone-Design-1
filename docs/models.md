# Model Documentation

> Comprehensive reference for all ML/DL models in the E-Commerce Customer Churn Prediction and Retention Optimization System.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Churn Prediction Models](#2-churn-prediction-models)
   - 2.1 [XGBoost](#21-xgboost)
   - 2.2 [LightGBM](#22-lightgbm)
   - 2.3 [LSTM](#23-lstm-long-short-term-memory)
   - 2.4 [Transformer](#24-transformer)
   - 2.5 [Ensemble](#25-ensemble-weighted-average)
3. [Uplift Modeling](#3-uplift-modeling)
   - 3.1 [T-Learner](#31-t-learner-two-model-approach)
   - 3.2 [S-Learner](#32-s-learner-single-model-approach)
   - 3.3 [Uplift Segmentation](#33-uplift-segmentation-4-quadrant)
4. [CLV Prediction](#4-clv-prediction)
   - 4.1 [Current Implementation: ML-based CLV Regression](#41-current-implementation-ml-based-clv-regression)
   - 4.2 [Implementation Notes](#42-implementation-notes)
5. [Survival Analysis](#5-survival-analysis)
   - 5.1 [Kaplan-Meier Estimator](#51-kaplan-meier-estimator)
   - 5.2 [Cox Proportional Hazards](#52-cox-proportional-hazards)
6. [Training Pipelines](#6-training-pipelines)
7. [Feature Engineering](#7-feature-engineering)
8. [MLflow Tracking Integration](#8-mlflow-tracking-integration)
9. [Model Serving & Real-Time Scoring](#9-model-serving--real-time-scoring)
10. [Reproducibility](#10-reproducibility)

---

## 1. System Overview

### Architecture

The system implements a multi-model pipeline for predicting customer churn, estimating treatment effects (uplift), forecasting customer lifetime value (CLV), and analyzing time-to-churn (survival). All models are orchestrated through a unified pipeline with three execution modes:

| Mode | Command | Description |
|------|---------|-------------|
| `train` | `--mode train` | Train churn prediction (ML + DL + Ensemble), CLV, and monitoring models |
| `uplift` | `--mode uplift` | Train uplift models (T-Learner, S-Learner), segment customers |
| `optimize` | `--mode optimize --budget 50000000` | Budget-constrained retention optimization via LP |

### Model Hierarchy

```
Churn Prediction (Primary)
├── ML Models (weight: 0.6)
│   ├── XGBoost (gradient boosting)
│   └── LightGBM (DART booster)
├── DL Models (weight: 0.4)
│   ├── LSTM (sequence model)
│   └── Transformer (attention model)
└── Ensemble (weighted average)

Causal Inference (Uplift)
├── T-Learner (two separate models)
└── S-Learner (single model with treatment indicator)

CLV Prediction
├── ML-based CLV regression (current)
└── BG/NBD + Gamma-Gamma (optional extension)

Survival Analysis (Bonus)
├── Kaplan-Meier (non-parametric)
└── Cox Proportional Hazards (semi-parametric)
```

### Data Split Strategy

- **Time-based split**: 10 months training / 2 months testing (configurable in `config/simulator_config.yaml`)
- **Training period**: 2024-01-01 to 2024-10-31
- **Test period**: 2024-11-01 to 2024-12-31
- **Cross-validation**: 5-fold CV on training data for ML models
- **Early stopping**: Validation-based for DL models

---

## 2. Churn Prediction Models

### Churn Definition

A customer is labeled as **churned** if either condition is met (configurable `OR` operator):

| Condition | Default Threshold | Config Key |
|-----------|-------------------|------------|
| No purchase | 30 days | `churn_definition.no_purchase_days` |
| No login/visit | 60 days | `churn_definition.no_login_days` |

Target churn rate: 15–25% of the customer population.

---

### 2.1 XGBoost

**Type**: Gradient Boosted Decision Trees
**Library**: `xgboost`
**Source**: `src/models/ml_models.py`

#### Purpose

Primary ML model for binary churn classification. XGBoost excels at tabular data with its regularized boosting framework and handles class imbalance natively via `scale_pos_weight`.

#### Hyperparameters

| Parameter | Default | Search Range | Description |
|-----------|---------|-------------|-------------|
| `n_estimators` | 300 | 100–500 | Number of boosting rounds |
| `max_depth` | 6 | 5–10 | Maximum tree depth |
| `learning_rate` | 0.05 | 0.01–0.1 | Step size shrinkage |
| `min_child_weight` | 3 | 1–5 | Minimum sum of instance weight in a child |
| `subsample` | 0.8 | 0.7–1.0 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | 0.7–1.0 | Column subsampling ratio |
| `reg_alpha` | 0.1 | 0–1 | L1 regularization |
| `reg_lambda` | 1.0 | 0.5–2.0 | L2 regularization |
| `scale_pos_weight` | auto | — | Ratio of negative to positive samples |
| `eval_metric` | `auc` | — | Evaluation metric for early stopping |
| `objective` | `binary:logistic` | — | Binary classification objective |
| `random_state` | 42 | — | Random seed for reproducibility |

#### Training Process

1. Load engineered features (30+ features) from the file-based feature store
2. Apply time-based train/test split (10 months / 2 months)
3. Handle class imbalance via `scale_pos_weight` (auto-calculated from class distribution)
4. Run 5-fold cross-validation with stratified splits
5. Perform hyperparameter tuning (Grid Search or Optuna)
6. Train final model on full training set
7. Optimize classification threshold via Precision-Recall curve
8. Generate SHAP feature importance explanations

#### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | ≥ 0.78 | Area under ROC curve (primary metric) |
| Precision | ≥ 0.73 | Positive predictive value |
| Recall | ≥ 0.69 | Sensitivity / True positive rate |
| F1-Score | ≥ 0.71 | Harmonic mean of precision and recall |
| CV Mean ± Std | Reported | 5-fold cross-validation stability |

#### MLflow Tracking

```
Experiment: churn_prediction
Run Name: xgboost_{timestamp}
├── Parameters: all hyperparameters, cv_folds=5, feature_count
├── Metrics: auc_roc, precision, recall, f1_score, cv_mean, cv_std
├── Artifacts:
│   ├── model.pkl (serialized model)
│   ├── shap_summary.png (SHAP beeswarm plot)
│   ├── feature_importance.json (top features)
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── Tags: model_type=xgboost, stage=ml
```

---

### 2.2 LightGBM

**Type**: Gradient Boosted Decision Trees (DART booster)
**Library**: `lightgbm`
**Source**: `src/models/ml_models.py`

#### Purpose

Complementary ML model using LightGBM's DART (Dropouts meet Multiple Additive Regression Trees) booster for improved generalization. Often slightly outperforms XGBoost on this task due to leaf-wise growth and dropout regularization.

#### Hyperparameters

| Parameter | Default | Search Range | Description |
|-----------|---------|-------------|-------------|
| `n_estimators` | 300 | 100–500 | Number of boosting iterations |
| `max_depth` | -1 (unlimited) | 5–10 | Maximum tree depth |
| `learning_rate` | 0.05 | 0.01–0.1 | Step size shrinkage |
| `num_leaves` | 31 | 20–63 | Maximum leaves per tree |
| `min_child_samples` | 20 | 10–50 | Minimum data in a leaf |
| `subsample` | 0.8 | 0.7–1.0 | Bagging fraction |
| `colsample_bytree` | 0.8 | 0.7–1.0 | Feature fraction |
| `reg_alpha` | 0.1 | 0–1 | L1 regularization |
| `reg_lambda` | 1.0 | 0.5–2.0 | L2 regularization |
| `boosting_type` | `dart` | `gbdt`, `dart` | Booster type |
| `class_weight` | `balanced` | — | Auto-balance class weights |
| `objective` | `binary` | — | Binary classification |
| `metric` | `auc` | — | Evaluation metric |
| `random_state` | 42 | — | Random seed |

#### Training Process

1. Same data pipeline as XGBoost (shared feature store)
2. Apply `class_weight='balanced'` for imbalance handling
3. Run 5-fold stratified cross-validation
4. Hyperparameter tuning via Optuna or Grid Search
5. Train final model, extract feature importance
6. Threshold optimization on validation set

#### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | ≥ 0.78 | Primary ranking metric |
| Precision | ≥ 0.73 | Correctly identified churners |
| Recall | ≥ 0.69 | Proportion of churners caught |
| F1-Score | ≥ 0.71 | Balance between precision and recall |

#### MLflow Tracking

```
Experiment: churn_prediction
Run Name: lightgbm_{timestamp}
├── Parameters: all hyperparameters, boosting_type, cv_folds
├── Metrics: auc_roc, precision, recall, f1_score, cv_mean, cv_std
├── Artifacts:
│   ├── model.pkl (serialized model)
│   ├── shap_summary.png
│   ├── feature_importance.json
│   └── lgbm_feature_importance.png (native plot)
└── Tags: model_type=lightgbm, stage=ml
```

---

### 2.3 LSTM (Long Short-Term Memory)

**Type**: Recurrent Neural Network (Sequence Model)
**Library**: PyTorch 2.0+
**Source**: `src/models/dl_models.py`

#### Purpose

Captures temporal patterns in customer event sequences. LSTM processes the ordered sequence of customer actions (page views, searches, purchases, etc.) to detect behavioral patterns indicative of churn, such as decreasing engagement over time.

#### Architecture

```
Input (batch_size, seq_len, num_event_types)
    │
    ▼
Embedding Layer (num_event_types → embedding_dim)
    │
    ▼
LSTM Layer(s) (embedding_dim → hidden_size, num_layers, bidirectional=False)
    │
    ▼
Dropout (p=dropout_rate)
    │
    ▼
Fully Connected (hidden_size → 1)
    │
    ▼
Sigmoid Activation → churn_probability
```

#### Hyperparameters

| Parameter | Default | Search Range | Description |
|-----------|---------|-------------|-------------|
| `sequence_length` | 50 | 30–100 | Maximum sequence length (padded/truncated) |
| `embedding_dim` | 64 | 32–128 | Event type embedding dimension |
| `hidden_size` | 128 | 64–256 | LSTM hidden state size |
| `num_layers` | 2 | 1–3 | Number of stacked LSTM layers |
| `dropout` | 0.3 | 0.2–0.5 | Dropout probability |
| `learning_rate` | 0.001 | 0.0001–0.01 | Adam optimizer LR |
| `batch_size` | 64 | 32–128 | Training batch size |
| `max_epochs` | 100 | — | Maximum training epochs |
| `early_stopping_patience` | 10 | 5–15 | Epochs without improvement before stopping |
| `optimizer` | Adam | — | Optimizer type |
| `loss_function` | BCEWithLogitsLoss | — | Binary cross-entropy with logits |

#### Input Preprocessing

1. **Event sequence extraction**: Customer events ordered chronologically
2. **Event type encoding**: Each event type mapped to integer ID (8 event types)
3. **Padding/truncation**: Sequences padded to `sequence_length` (zero-padding at start)
4. **Feature augmentation**: Optional timestamp-derived features (hour, day-of-week)

#### Training Process

1. Convert tabular + event data into sequences per customer
2. Create PyTorch `Dataset` and `DataLoader` with padding
3. Train with Adam optimizer and BCEWithLogitsLoss
4. Monitor validation AUC-ROC for early stopping
5. Save best model checkpoint (lowest validation loss)
6. Evaluate on test set

#### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | ≥ 0.75 | Sequence-based prediction quality |
| Precision | ≥ 0.70 | Positive predictive value |
| Recall | ≥ 0.68 | Churner detection rate |
| F1-Score | ≥ 0.69 | Balanced metric |
| Training Loss | Converging | BCE loss trajectory |
| Validation Loss | Non-increasing | Early stopping criterion |

#### MLflow Tracking

```
Experiment: churn_prediction
Run Name: lstm_{timestamp}
├── Parameters: sequence_length, embedding_dim, hidden_size, num_layers,
│               dropout, learning_rate, batch_size, epochs_trained
├── Metrics: auc_roc, precision, recall, f1_score,
│            train_loss (per epoch), val_loss (per epoch)
├── Artifacts:
│   ├── model.pth (PyTorch state_dict)
│   ├── training_curves.png (loss/AUC over epochs)
│   └── architecture.txt (model summary)
└── Tags: model_type=lstm, stage=dl, framework=pytorch
```

---

### 2.4 Transformer

**Type**: Attention-Based Sequence Model
**Library**: PyTorch 2.0+
**Source**: `src/models/dl_models.py`

#### Purpose

Leverages multi-head self-attention to capture long-range dependencies in customer behavior sequences. The Transformer can identify complex patterns like "customer who searched frequently but stopped adding to cart 3 weeks ago" without the recurrence bottleneck of LSTMs.

#### Architecture

```
Input (batch_size, seq_len, num_event_types)
    │
    ▼
Embedding Layer (num_event_types → d_model)
    │
    ▼
Positional Encoding (sinusoidal or learnable)
    │
    ▼
TransformerEncoder (num_layers × TransformerEncoderLayer)
    ├── Multi-Head Self-Attention (nhead heads)
    ├── Feed-Forward Network (d_model → dim_feedforward → d_model)
    └── Layer Normalization + Dropout
    │
    ▼
Global Average Pooling (seq_len → 1)
    │
    ▼
Fully Connected (d_model → 1)
    │
    ▼
Sigmoid Activation → churn_probability
```

#### Hyperparameters

| Parameter | Default | Search Range | Description |
|-----------|---------|-------------|-------------|
| `sequence_length` | 50 | 30–100 | Maximum sequence length |
| `d_model` | 64 | 32–128 | Model/embedding dimension |
| `nhead` | 4 | 2–8 | Number of attention heads |
| `num_encoder_layers` | 2 | 1–4 | Number of Transformer encoder layers |
| `dim_feedforward` | 256 | 128–512 | Feed-forward network hidden size |
| `dropout` | 0.3 | 0.2–0.5 | Dropout probability |
| `learning_rate` | 0.0005 | 0.0001–0.005 | Adam optimizer LR |
| `batch_size` | 64 | 32–128 | Training batch size |
| `max_epochs` | 100 | — | Maximum epochs |
| `early_stopping_patience` | 10 | 5–15 | Early stopping patience |
| `warmup_steps` | 500 | — | Learning rate warmup steps |

#### Training Process

1. Same sequence preprocessing as LSTM
2. Add positional encoding to embedding output
3. Apply attention mask for padded positions
4. Train with Adam optimizer and BCEWithLogitsLoss
5. Optional learning rate warmup schedule
6. Early stopping on validation AUC-ROC
7. Save best checkpoint

#### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | ≥ 0.75 | Attention-based prediction quality |
| Precision | ≥ 0.70 | Positive predictive value |
| Recall | ≥ 0.68 | Churner detection rate |
| F1-Score | ≥ 0.69 | Balanced metric |

#### MLflow Tracking

```
Experiment: churn_prediction
Run Name: transformer_{timestamp}
├── Parameters: d_model, nhead, num_encoder_layers, dim_feedforward,
│               dropout, learning_rate, sequence_length, epochs_trained
├── Metrics: auc_roc, precision, recall, f1_score,
│            train_loss (per epoch), val_loss (per epoch)
├── Artifacts:
│   ├── model.pth (PyTorch state_dict)
│   ├── training_curves.png
│   ├── attention_weights.png (sample attention visualization)
│   └── architecture.txt
└── Tags: model_type=transformer, stage=dl, framework=pytorch
```

---

### 2.5 Ensemble (Weighted Average)

**Type**: Model Combination
**Source**: `src/models/ensemble.py`

#### Purpose

Combines ML and DL model predictions using a configurable weighted average to leverage the complementary strengths of tree-based models (tabular feature interactions) and sequence models (temporal patterns).

#### Ensemble Strategy

```
P_ensemble = w_ml × P_ml + w_dl × P_dl
```

Where:
- `P_ml` = Best ML model prediction (max of XGBoost, LightGBM AUC)
- `P_dl` = Best DL model prediction (max of LSTM, Transformer AUC)
- `w_ml` = 0.6 (configurable via `pipeline.ensemble_weight_ml`)
- `w_dl` = 0.4 (configurable via `pipeline.ensemble_weight_dl`)

#### Configuration

```yaml
# config/simulator_config.yaml
pipeline:
  ensemble_weight_ml: 0.6
  ensemble_weight_dl: 0.4
```

#### Model Selection Logic

1. Train all 4 models (XGBoost, LightGBM, LSTM, Transformer)
2. Select best ML model by validation AUC-ROC
3. Select best DL model by validation AUC-ROC
4. Combine using weighted average
5. Optimize ensemble threshold on validation set

#### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AUC-ROC | ≥ 0.78 (system min) | Combined model performance |
| Precision | ≥ 0.76 | Ensemble positive predictive value |
| Recall | ≥ 0.74 | Ensemble churner detection |
| F1-Score | ≥ 0.75 | Ensemble balanced metric |

#### MLflow Tracking

```
Experiment: churn_prediction
Run Name: ensemble_{timestamp}
├── Parameters: ml_weight, dl_weight, best_ml_model, best_dl_model,
│               ml_auc, dl_auc, threshold
├── Metrics: auc_roc, precision, recall, f1_score
├── Artifacts:
│   ├── ensemble_predictions.csv
│   ├── roc_comparison.png (all models + ensemble)
│   └── confusion_matrix.png
└── Tags: model_type=ensemble, stage=final
```

---

## 3. Uplift Modeling

### Overview

Uplift modeling estimates the **causal effect** of marketing interventions on individual customers. Rather than predicting "will this customer churn?", uplift models answer "will this marketing action **prevent** this customer from churning?"

**Treatment/Control split**: 50/50 (configurable via `treatment.treatment_ratio`)
**Minimum group size**: 10,000 per group

---

### 3.1 T-Learner (Two-Model Approach)

**Type**: Meta-Learner for Heterogeneous Treatment Effects
**Source**: `src/uplift/uplift_model.py`

#### How It Works

The T-Learner trains **two separate models**:

1. **Treatment model** `M_T`: Trained on customers who received marketing (treatment group)
2. **Control model** `M_C`: Trained on customers who did not receive marketing (control group)

**Individual Treatment Effect (ITE)**:
```
τ(x) = M_T(x) - M_C(x)
```

Where `τ(x)` is the **Conditional Average Treatment Effect (CATE)** — the estimated uplift for customer `x`.

#### Base Learner

- **Algorithm**: XGBoost or LightGBM (same hyperparameters as churn models)
- **Target**: Binary churn label within each group
- **Features**: Same 30+ features from the feature store

#### Training Process

1. Split data into treatment and control groups
2. Train `M_T` on treatment group: `X_treatment → y_treatment`
3. Train `M_C` on control group: `X_control → y_control`
4. For each customer: `uplift_score = M_T.predict_proba(x) - M_C.predict_proba(x)`
5. Positive uplift → marketing reduces churn probability
6. Negative uplift → marketing increases churn probability ("sleeping dogs")

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Qini Score | Area between Qini curve and random baseline |
| Qini AUC | Normalized Qini coefficient |
| Average Treatment Effect (ATE) | Mean uplift across population |
| CATE Distribution | Histogram of individual treatment effects |
| Cumulative Gain | Incremental conversions vs. population targeted |

#### MLflow Tracking

```
Experiment: uplift_modeling
Run Name: t_learner_{timestamp}
├── Parameters: base_learner, treatment_ratio, n_treatment, n_control
├── Metrics: qini_score, qini_auc, avg_treatment_effect,
│            treatment_model_auc, control_model_auc
├── Artifacts:
│   ├── treatment_model.pkl
│   ├── control_model.pkl
│   ├── qini_curve.png
│   ├── uplift_distribution.png
│   └── cate_by_segment.json
└── Tags: model_type=t_learner, stage=uplift
```

---

### 3.2 S-Learner (Single-Model Approach)

**Type**: Meta-Learner for Heterogeneous Treatment Effects
**Source**: `src/uplift/uplift_model.py`

#### How It Works

The S-Learner trains a **single model** with the treatment indicator as an additional feature:

```
M(X, T) → y
```

**Individual Treatment Effect (ITE)**:
```
τ(x) = M(x, T=1) - M(x, T=0)
```

#### Advantages Over T-Learner

- More sample-efficient (uses all data for one model)
- Better when treatment effect is small
- Simpler to deploy

#### Disadvantages

- May underestimate treatment effect if model ignores treatment indicator
- Less flexible for complex treatment interactions

#### Training Process

1. Combine treatment and control data
2. Add binary `treatment_indicator` column as feature
3. Train single model: `(X, T) → y`
4. For each customer:
   - Predict with `T=1`: `P_treated = M(x, T=1)`
   - Predict with `T=0`: `P_control = M(x, T=0)`
   - `uplift = P_treated - P_control`

#### MLflow Tracking

```
Experiment: uplift_modeling
Run Name: s_learner_{timestamp}
├── Parameters: base_learner, treatment_ratio, feature_count
├── Metrics: qini_score, qini_auc, avg_treatment_effect, model_auc
├── Artifacts:
│   ├── model.pkl
│   ├── qini_curve.png
│   └── uplift_distribution.png
└── Tags: model_type=s_learner, stage=uplift
```

---

### 3.3 Uplift Segmentation (4-Quadrant)

Based on the combination of **churn probability** and **uplift score**, customers are classified into four quadrants:

| Segment | Condition | Strategy |
|---------|-----------|----------|
| **Persuadables** | Uplift > 0.05 AND Churn Prob > 0.5 | High-priority targeting — marketing is effective |
| **Sure Things** | Uplift ≤ 0.05 AND Churn Prob ≤ 0.5 | No action needed — loyal without intervention |
| **Lost Causes** | Uplift ≤ 0.05 AND Churn Prob > 0.5 | Deprioritize — marketing has no effect |
| **Sleeping Dogs** | Uplift < 0 | Do NOT target — marketing causes churn |

#### Extended 6-Segment Classification

For budget optimization, customers are further segmented using **churn probability × uplift score × CLV**:

| Segment | Priority | Budget Allocation |
|---------|----------|-------------------|
| High CLV + Persuadable | Highest | Premium retention actions |
| High CLV + Sure Thing | Medium | Light engagement maintenance |
| High CLV + Lost Cause | Low | Minimal — analyze root causes |
| Low CLV + Persuadable | Medium | Cost-effective retention |
| Low CLV + Sure Thing | Low | No action |
| Low CLV + Lost Cause | None | No budget allocated |

---

## 4. CLV Prediction

### Overview

Customer Lifetime Value (CLV) prediction estimates the total revenue a customer will generate over the next 12 months. This feeds into the budget optimization module to prioritize retention spending on high-value customers.

---

### 4.1 Current Implementation: ML-based CLV Regression

**Type**: Supervised regression
**Library**: scikit-learn / gradient boosting stack
**Source**: `src/models/clv_model.py`

#### How It Works

The repository implements the requirement's ML-based option rather than a `lifetimes` BG/NBD package. The pipeline trains a CLV regressor from engineered behavioral and RFM features, validates it on a holdout split, then refits on all available proxy labels before writing customer-level predictions.

**Core inputs**:
- **Frequency**: Number of repeat purchases
- **Recency**: Days since recent activity or purchase
- **Monetary**: Total or average order value (KRW)
- **Behavioral features**: session quality, trend, timing, journey, and sequence features

**CLV Formula**:
```
CLV = model.predict(engineered_features), using a 12-month monetary proxy label
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prediction_horizon` | 12 months | CLV forecast period |
| `observation_period` | 12 months | Historical data window |
| `discount_rate` | 0.01 | Monthly discount rate |
| `profit_margin` | 0.30 | Estimated profit margin |
| `high_value_percentile` | 80th percentile | Top-20% high-value flag |

#### Training Process

1. Calculate RFM (Recency, Frequency, Monetary) metrics per customer
2. Build the engineered feature matrix
3. Train on the first 80% of customers and validate on a 20% holdout
4. Save actual-vs-predicted validation rows and MAE/RMSE/R² summary
5. Refit on the full dataset for final customer-level predictions
6. Identify top-20% high-value customers

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error on holdout period |
| MAE | Mean absolute error |
| R² | Coefficient of determination |
| Top-20% Flag | Customers above the 80th percentile predicted CLV |

#### MLflow Tracking

```
Experiment: clv_prediction
Run Name: ml_clv_{timestamp}
├── Parameters: prediction_horizon, holdout_ratio, feature_set
├── Metrics: rmse, mae, r2_score
├── Artifacts:
│   ├── clv_model.pkl
│   ├── clv_predictions.csv
│   ├── clv_validation.json
│   ├── clv_actual_vs_predicted.csv
│   └── clv_top_customers.csv
└── Tags: model_type=ml_clv, stage=clv
```

---

### 4.2 Implementation Notes

BG/NBD + Gamma-Gamma is a valid future extension, but it is not the code path currently used by `python src/main.py --mode clv`.

#### How It Works

An alternative approach using gradient boosted trees to directly predict CLV as a regression target, incorporating richer features beyond RFM.

#### Features

- All 30+ engineered features from the churn model
- RFM metrics (recency, frequency, monetary)
- Customer tenure, purchase velocity
- Product category diversity
- Seasonal patterns, weekend activity ratio
- Session quality metrics

#### Hyperparameters

Same as XGBoost churn model (Section 2.1), with the following changes:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `objective` | `reg:squarederror` | Regression objective |
| `eval_metric` | `rmse` | Root mean squared error |

#### MLflow Tracking

```
Experiment: clv_prediction
Run Name: ml_clv_{timestamp}
├── Parameters: same as XGBoost churn + objective=regression
├── Metrics: rmse, mae, r2_score, mape
├── Artifacts:
│   ├── clv_model.pkl
│   ├── shap_clv_importance.png
│   └── predicted_vs_actual.png
└── Tags: model_type=ml_clv, stage=clv
```

---

## 5. Survival Analysis

### Overview

Survival analysis models the **time-to-churn** event, providing richer insights than binary churn prediction: "When will this customer likely churn?" rather than just "Will they churn?"

---

### 5.1 Kaplan-Meier Estimator

**Type**: Non-Parametric Survival Estimator
**Library**: `lifelines`
**Source**: `src/models/survival_analysis.py`

#### How It Works

The Kaplan-Meier estimator produces a **survival function** S(t) — the probability that a customer has not yet churned by time t. It handles right-censored data (customers who haven't churned by end of observation).

#### Key Outputs

| Output | Description |
|--------|-------------|
| Survival Curve | S(t) plot showing churn probability over time |
| Median Survival Time | Time at which 50% of customers have churned |
| Per-Segment Curves | Survival curves stratified by persona/segment |
| Confidence Intervals | 95% CI bands around survival estimates |

#### Training Process

1. Define duration: days from first purchase to churn event (or censoring)
2. Define event: binary indicator (1 = churned, 0 = censored/still active)
3. Fit Kaplan-Meier estimator on (duration, event)
4. Optionally stratify by customer segment
5. Perform log-rank test between segments

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Median Survival | Days until 50% churn probability |
| Log-Rank p-value | Significance of between-group differences (< 0.05) |
| Survival at 30/60/90 days | Retention probability at key milestones |

#### MLflow Tracking

```
Experiment: survival_analysis
Run Name: kaplan_meier_{timestamp}
├── Parameters: segments_analyzed, confidence_level
├── Metrics: median_survival_days, logrank_pvalue,
│            survival_30d, survival_60d, survival_90d
├── Artifacts:
│   ├── survival_curves.png
│   ├── per_segment_curves.png
│   └── survival_table.csv
└── Tags: model_type=kaplan_meier, stage=survival
```

---

### 5.2 Cox Proportional Hazards

**Type**: Semi-Parametric Regression Model
**Library**: `lifelines`
**Source**: `src/models/survival_analysis.py`

#### How It Works

The Cox PH model estimates the effect of covariates (features) on the **hazard rate** (instantaneous risk of churning). It produces hazard ratios that quantify how each feature increases or decreases churn risk.

**Hazard Function**:
```
h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₚXₚ)
```

Where:
- `h₀(t)` = baseline hazard (non-parametric)
- `βᵢ` = log-hazard ratio for feature i
- `exp(βᵢ)` = hazard ratio: >1 increases risk, <1 decreases risk

#### Features Used

A subset of the 30+ engineered features selected for survival modeling:

- RFM features (recency, frequency, monetary)
- Behavioral change features (visit_frequency_change, purchase_cycle_change)
- Session quality features (avg_session_duration, pageviews_per_session)
- Journey stage features
- Customer tenure

#### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `penalizer` | 0.01 | L2 regularization strength |
| `l1_ratio` | 0.0 | ElasticNet mixing (0=L2, 1=L1) |
| `baseline_estimation_method` | `breslow` | Baseline hazard estimator |
| `step_size` | 0.5 | Newton-Raphson step size |

#### Training Process

1. Prepare duration and event columns
2. Select features, handle multicollinearity (VIF check)
3. Standardize continuous features
4. Fit CoxPH model with penalization
5. Check proportional hazards assumption (Schoenfeld residuals)
6. Extract hazard ratios and confidence intervals

#### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Concordance Index | ≥ 0.812 | Probability of correctly ranking pairs |
| Log-Partial Likelihood | — | Model fit statistic |
| AIC | — | Model selection criterion |
| Schoenfeld p-value | > 0.05 | PH assumption validity per covariate |

#### Key Outputs

| Output | Description |
|--------|-------------|
| Hazard Ratios | Per-feature risk multipliers with 95% CI |
| Baseline Survival | S₀(t) for average customer |
| Individual Risk Scores | `exp(Σ βᵢXᵢ)` per customer |
| Risk Stratification | High/Medium/Low risk groups |

#### MLflow Tracking

```
Experiment: survival_analysis
Run Name: cox_ph_{timestamp}
├── Parameters: penalizer, l1_ratio, features_used, n_covariates
├── Metrics: concordance_index, log_partial_likelihood, aic,
│            ph_assumption_pvalue
├── Artifacts:
│   ├── cox_model.pkl
│   ├── hazard_ratios.png (forest plot)
│   ├── hazard_ratios.json (coefficients)
│   ├── schoenfeld_residuals.png
│   └── baseline_survival.png
└── Tags: model_type=cox_ph, stage=survival
```

---

## 6. Training Pipelines

### Pipeline Modes

#### Mode 1: `train` — Full Model Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Data Generation                                             │
│   CustomerDataGenerator → 20,000 customers, 12 months, 8 events    │
│   Output: raw customer data, event logs                             │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: Cohort & Journey Analysis                                   │
│   Monthly cohort retention curves, customer journey funnels         │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: Feature Engineering                                         │
│   30+ features: RFM, behavioral change, session quality, sequence   │
│   Output: feature_store/ (Parquet files)                            │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: ML Model Training                                           │
│   XGBoost + LightGBM with 5-fold CV, SHAP explanations              │
├─────────────────────────────────────────────────────────────────────┤
│ Step 5: DL Model Training                                           │
│   LSTM + Transformer with early stopping, PyTorch                   │
├─────────────────────────────────────────────────────────────────────┤
│ Step 6: Ensemble Creation                                           │
│   Weighted average: 0.6 × best_ML + 0.4 × best_DL                  │
├─────────────────────────────────────────────────────────────────────┤
│ Step 7: CLV Prediction                                              │
│   ML-based regressor → 12-month CLV per customer                   │
├─────────────────────────────────────────────────────────────────────┤
│ Step 8: Model Monitoring                                            │
│   PSI + KS-test for data drift detection                            │
├─────────────────────────────────────────────────────────────────────┤
│ Step 9: MLflow Logging                                              │
│   All metrics, parameters, and artifacts tracked                    │
├─────────────────────────────────────────────────────────────────────┤
│ Step 10: Checkpoint                                                 │
│   pipeline_state.json updated (completed/failed/pending)            │
└─────────────────────────────────────────────────────────────────────┘
```

#### Mode 2: `uplift` — Uplift Modeling Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Load trained models + feature store                         │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: Train T-Learner (treatment + control models)                │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: Train S-Learner (single model with treatment indicator)     │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: Calculate CATE (uplift scores) for all customers            │
├─────────────────────────────────────────────────────────────────────┤
│ Step 5: 4-Quadrant segmentation (Persuadable/Sure/Lost/Sleeping)    │
├─────────────────────────────────────────────────────────────────────┤
│ Step 6: 6-Segment classification (churn × uplift × CLV)             │
├─────────────────────────────────────────────────────────────────────┤
│ Step 7: Generate Qini curves and visualizations                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Mode 3: `optimize` — Budget Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Load uplift scores + CLV predictions                        │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: Formulate LP: max Σ(Uplift_i × CLV_i × Action_i)           │
│         subject to: Σ(Cost_i × Action_i) ≤ Budget                  │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: Solve LP with budget = 50,000,000 KRW                      │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: What-if analysis (50%, 100%, 200% budget scenarios)         │
├─────────────────────────────────────────────────────────────────────┤
│ Step 5: A/B test design with power analysis                         │
├─────────────────────────────────────────────────────────────────────┤
│ Step 6: Statistical significance testing                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Checkpoint System

State is persisted to `pipeline_state.json`:

```json
{
  "pipeline_id": "run_20240101_120000",
  "status": "completed",
  "steps": {
    "data_generation": {"status": "completed", "duration_sec": 45},
    "feature_engineering": {"status": "completed", "duration_sec": 120},
    "ml_training": {"status": "completed", "duration_sec": 300},
    "dl_training": {"status": "completed", "duration_sec": 600},
    "ensemble": {"status": "completed", "duration_sec": 10},
    "clv_prediction": {"status": "completed", "duration_sec": 60},
    "monitoring": {"status": "completed", "duration_sec": 30}
  },
  "timestamp": "2024-01-01T12:00:00",
  "random_seed": 42
}
```

Each step can have status: `pending`, `running`, `completed`, or `failed`. On restart, the pipeline resumes from the last non-completed step.

---

## 7. Feature Engineering

### Feature Groups (30+ Total)

All features are documented in detail in `docs/feature_dictionary.md`. Summary:

| Group | Count | Examples |
|-------|-------|---------|
| RFM | 3 | recency, frequency, monetary |
| Behavioral Change | 5+ | visit_frequency_change, purchase_cycle_change, session_duration_change, cart_conversion_change, coupon_response_change |
| Purchase Cycle Anomaly | 1 | purchase_cycle_anomaly (days_since / avg_cycle) |
| Session Quality | 3+ | avg_session_duration, pageviews_per_session, search_to_purchase_rate |
| Sequence Features | 2+ | event_sequence_embedding, behavior_cluster_id |
| Time-based | 2+ | weekend_purchase_ratio, time_of_day_ratios |
| Journey Stage | 2+ | journey_stage, stage_tenure_days |
| Interaction Features | Variable | Cross-feature products and ratios |

### Preprocessing

- **Missing values**: Filled with mean/median/forward-fill (no NaN allowed)
- **Infinite values**: Clipped to feature-specific bounds
- **Outliers**: Handled via IQR method or percentile capping
- **Scaling**: StandardScaler for ML models, MinMaxScaler for DL models
- **Sequence padding**: Zero-padding to `sequence_length` for DL models

---

## 8. MLflow Tracking Integration

### Server Configuration

| Setting | Value |
|---------|-------|
| Server URL | `http://localhost:5001` from host, `http://mlflow:5000` inside Docker |
| Backend Store | SQLite (`mlruns/mlruns.db`) |
| Artifact Store | Local directory (`mlruns/artifacts/`) |
| Container | `mlflow` service in Docker Compose |

### Experiment Organization

```
MLflow Server (host http://localhost:5001, Docker http://mlflow:5000)
├── Experiment: churn_prediction
│   ├── Run: xgboost_{timestamp}
│   ├── Run: lightgbm_{timestamp}
│   ├── Run: lstm_{timestamp}
│   ├── Run: transformer_{timestamp}
│   └── Run: ensemble_{timestamp}
│
├── Experiment: uplift_modeling
│   ├── Run: t_learner_{timestamp}
│   └── Run: s_learner_{timestamp}
│
├── Experiment: clv_prediction
│   └── Run: ml_clv_{timestamp}
│
└── Experiment: survival_analysis
    ├── Run: kaplan_meier_{timestamp}
    └── Run: cox_ph_{timestamp}
```

### What Gets Tracked

For every model run, MLflow captures:

1. **Parameters**: All hyperparameters, data split info, feature counts, random seeds
2. **Metrics**: Primary evaluation metrics (AUC, precision, recall, F1, RMSE, etc.)
3. **Artifacts**: Serialized models, plots (SHAP, ROC, confusion matrix, Qini curves), JSON configs
4. **Tags**: `model_type`, `stage`, `framework`, `pipeline_id`
5. **System Metrics**: Training duration, memory usage

### Integration Code Pattern

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("churn_prediction")

with mlflow.start_run(run_name=f"xgboost_{timestamp}"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
    })

    # Train model
    model = train_xgboost(X_train, y_train)

    # Log metrics
    metrics = evaluate(model, X_test, y_test)
    mlflow.log_metrics({
        "auc_roc": metrics["auc_roc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    })

    # Log artifacts
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("shap_summary.png")
```

### Model Registry

- Best-performing models are registered in MLflow Model Registry
- Stages: `None` → `Staging` → `Production` → `Archived`
- The ensemble model (production) is the primary model served for real-time scoring

---

## 9. Model Serving & Real-Time Scoring

### Architecture

Real-time scoring uses Redis as the feature store and message broker:

```
Customer Event → Redis Stream → Real-Time Scorer → Churn Score → Redis Cache
                                       │
                                       ▼
                              MLflow Model Registry
                              (Load production model)
```

### Scoring Latency Target

| Component | Target | Description |
|-----------|--------|-------------|
| Feature lookup | < 10ms | Redis GET for cached features |
| Model inference | < 50ms | Ensemble prediction |
| Total latency | < 100ms | End-to-end scoring |

### Redis Schema

```
Key: customer:{customer_id}:features  → Hash (feature values)
Key: customer:{customer_id}:score     → String (churn probability)
Key: customer:{customer_id}:segment   → String (uplift segment)
Stream: churn_events                  → Stream (real-time events)
```

---

## 10. Reproducibility

### Configuration for Reproducibility

All sources of randomness are controlled via `random_seed: 42` in `config/simulator_config.yaml`:

| Component | Seed Control |
|-----------|-------------|
| Data generation | `np.random.seed(42)` |
| Train/test split | `random_state=42` in sklearn |
| XGBoost | `random_state=42` |
| LightGBM | `random_state=42` |
| PyTorch | `torch.manual_seed(42)`, `torch.backends.cudnn.deterministic=True` |
| Cross-validation | `StratifiedKFold(random_state=42)` |

### Deterministic Guarantees

With the same seed and configuration:
- Identical synthetic data generated
- Identical train/test splits
- Identical model weights (CPU only — no GPU non-determinism)
- Identical evaluation metrics
- Identical MLflow logged results

### Configuration Management

All parameters are managed via YAML files in the `config/` folder:

| File | Contents |
|------|----------|
| `config/simulator_config.yaml` | Simulation, churn definition, personas, feature/model/uplift/budget/dashboard settings |

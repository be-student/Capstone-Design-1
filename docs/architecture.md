# System Architecture

> **E-Commerce Customer Churn Prediction & Retention Optimization System**

This document describes the system architecture, component relationships, data flow, and module responsibilities for the end-to-end churn prediction and retention optimization platform.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component Diagram](#3-component-diagram)
4. [Data Flow](#4-data-flow)
5. [Module Descriptions](#5-module-descriptions)
   - 5.1 [Data Layer](#51-data-layer)
   - 5.2 [Feature Engineering](#52-feature-engineering)
   - 5.3 [Model Layer](#53-model-layer)
   - 5.4 [Optimization Layer](#54-optimization-layer)
   - 5.5 [Analysis Layer](#55-analysis-layer)
   - 5.6 [Monitoring Layer](#56-monitoring-layer)
   - 5.7 [Streaming Layer](#57-streaming-layer)
   - 5.8 [Pipeline Orchestration](#58-pipeline-orchestration)
   - 5.9 [Dashboard Layer](#59-dashboard-layer)
6. [Infrastructure & Deployment](#6-infrastructure--deployment)
7. [Configuration Management](#7-configuration-management)
8. [Testing Architecture](#8-testing-architecture)

---

## 1. Architecture Overview

The system is a modular ML/DL platform that predicts customer churn, estimates customer lifetime value, models uplift from retention interventions, optimizes marketing budgets, and delivers actionable insights through an interactive dashboard. It follows a layered architecture with clear separation of concerns:

- **Data Layer** — Simulation, ingestion, and preprocessing
- **Feature Layer** — 30+ engineered features (RFM, behavioral, sequential)
- **Model Layer** — ML (XGBoost/LightGBM), DL (Transformer/LSTM), Ensemble, Uplift, CLV, Survival
- **Optimization Layer** — LP-based budget allocation across marketing channels
- **Analysis Layer** — A/B testing, cohort analysis, what-if scenarios
- **Monitoring Layer** — Drift detection (PSI, KS-test), alerting
- **Serving Layer** — Real-time scoring via Redis streams, REST API
- **Presentation Layer** — Streamlit dashboard with 10+ views

All components are containerized with Docker Compose and tracked via MLflow.

---

## 2. High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLI Entrypoint (src/main.py)                     │
│                    --mode: simulate | train | uplift | clv |             │
│                    optimize | ab_test | survival | recommend |           │
│                    cohort | monitor | dashboard | all                    │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────────┐
│                    Pipeline Orchestration (src/pipeline/)                 │
│              16-stage pipeline with checkpoint/resume support             │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────────┘
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Data │ │Feature │ │ Model  │ │Optimi- │ │Analysis│ │Monitor-│
│Layer │ │Engine- │ │ Layer  │ │zation  │ │ Layer  │ │  ing   │
│      │ │ering   │ │        │ │ Layer  │ │        │ │ Layer  │
└──┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
   │         │          │          │          │          │
   └─────────┴──────────┴──────┬───┴──────────┴──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  Streamlit   │ │   MLflow     │ │    Redis     │
     │  Dashboard   │ │  Tracking    │ │  Streaming   │
     │  :8501       │ │  :5000       │ │  :6379       │
     └──────────────┘ └──────────────┘ └──────────────┘
```

---

## 3. Component Diagram

```
src/
├── main.py                          # CLI entrypoint (14 modes)
├── __main__.py                      # python -m src support
│
├── data/                            # DATA LAYER
│   ├── generator.py                 #   CustomerDataGenerator (6 personas)
│   ├── preprocessing.py             #   Data cleaning & validation
│   └── orchestrator.py              #   Orchestrates data pipeline stages
│
├── features/                        # FEATURE LAYER
│   ├── feature_engineering.py       #   30+ features (RFM, behavioral, sequence)
│   └── segmentation.py             #   RFM-based & K-means segmentation
│
├── models/                          # MODEL LAYER
│   ├── churn_model.py               #   MLChurnModel, DLChurnModel, EnsembleChurnModel
│   ├── dl_trainer.py                #   PyTorch training loop (early stopping)
│   ├── sequence_utils.py            #   ChurnSequenceDataset for LSTM/Transformer
│   ├── shap_explainer.py            #   SHAP-based feature importance
│   ├── uplift_model.py              #   T-Learner / S-Learner uplift modeling
│   ├── clv_model.py                 #   Customer Lifetime Value (Gradient Boosting)
│   ├── survival_model.py            #   Kaplan-Meier & Cox PH survival analysis
│   ├── survival_analysis.py         #   Additional survival utilities
│   ├── budget_optimizer.py          #   LP-based budget allocation
│   ├── whatif_analysis.py           #   What-if scenario analyzer
│   ├── ab_testing.py                #   A/B test design & power analysis
│   ├── recommendations.py           #   Personalized retention actions
│   ├── scoring_api.py               #   Real-time scoring API
│   └── mlflow_tracking.py           #   MLflowTracker & ModelRegistry
│
├── optimization/                    # OPTIMIZATION LAYER
│   └── budget_optimizer.py          #   LP formulation (scipy.optimize.linprog)
│
├── analysis/                        # ANALYSIS LAYER
│   ├── ab_testing.py                #   ABTestAnalyzer (power, Chi-square, Z-test)
│   └── cohort_analysis.py           #   CohortAnalyzer (retention, revenue, churn)
│
├── monitoring/                      # MONITORING LAYER
│   ├── drift_detection.py           #   PSI-based drift detection
│   ├── ks_drift.py                  #   KS-test drift detection
│   └── monitoring_service.py        #   Monitoring orchestration & alerting
│
├── streaming/                       # STREAMING LAYER
│   ├── redis_producer.py            #   Publishes scoring requests to Redis
│   └── redis_consumer.py            #   Consumes & scores via Redis streams
│
├── pipeline/                        # ORCHESTRATION LAYER
│   ├── runner.py                    #   16-stage PipelineRunner
│   └── pipeline_state.py           #   Checkpoint state management
│
└── dashboard/                       # PRESENTATION LAYER
    ├── app.py                       #   Main Streamlit application
    ├── data_loader.py               #   Data loading & caching
    ├── monitoring_view.py           #   Drift & alert visualization
    ├── recommendations_view.py      #   Retention action recommendations
    ├── system_health_view.py        #   Pipeline & system health
    └── utils/
        └── dashboard_helpers.py     #   Formatting, colors, risk classification
```

---

## 4. Data Flow

The system processes data through a multi-stage pipeline. Each stage produces artifacts consumed by downstream stages.

### 4.1 End-to-End Pipeline Flow

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Stage 1: DATA GENERATION                                               │
 │   Input:  config/simulator_config.yaml (personas, event types)         │
 │   Output: customers.csv, events.csv (20K customers, 12 months)         │
 │   Module: src/data/generator.py                                        │
 └─────────────────────────────┬───────────────────────────────────────────┘
                               ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Stage 2: PREPROCESSING                                                  │
 │   Input:  Raw customers + events                                        │
 │   Output: Cleaned, validated, time-split data (train/test)              │
 │   Module: src/data/preprocessing.py                                     │
 └─────────────────────────────┬───────────────────────────────────────────┘
                               ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Stage 3: FEATURE ENGINEERING                                            │
 │   Input:  Cleaned customer + event data                                 │
 │   Output: 30+ features per customer (RFM, behavioral, session, seq.)    │
 │   Module: src/features/feature_engineering.py                           │
 └──────────┬──────────────────┬───────────────────────────────────────────┘
            │                  │
            ▼                  ▼
 ┌─────────────────┐  ┌──────────────────┐
 │ Stage 4: ML     │  │ Stage 5: DL      │
 │ XGBoost/LGBM    │  │ Transformer/LSTM │
 │ 5-Fold CV       │  │ PyTorch training │
 └────────┬────────┘  └────────┬─────────┘
          │                    │
          └────────┬───────────┘
                   ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Stage 6: ENSEMBLE CREATION                                              │
 │   Weighted average: 0.6 × ML + 0.4 × DL                                │
 │   Output: Churn probability per customer                                │
 └──────────┬──────────────────┬───────────────────────────────────────────┘
            │                  │
            ▼                  ▼
 ┌──────────────────┐  ┌───────────────────┐
 │ Stage 7: UPLIFT  │  │ Stage 8: CLV      │
 │ T/S-Learner      │  │ Gradient Boosting │
 │ CATE estimation  │  │ Customer value    │
 │ 4-quadrant segm. │  │ prediction        │
 └────────┬─────────┘  └────────┬──────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Stage 9: BUDGET OPTIMIZATION                                            │
 │   LP: maximize Σ(uplift_i × CLV_i × action_i)                          │
 │   Subject to: total budget ≤ 50M KRW, per-channel limits               │
 │   Channels: email, SMS, push, coupon, call center                       │
 └─────────────────────────────┬───────────────────────────────────────────┘
                               ▼
 ┌──────────┬──────────┬───────────────┬──────────────┐
 │Stage 10  │Stage 11  │  Stage 12     │  Stage 13    │
 │A/B Test  │Survival  │  Recommend-   │  Scoring     │
 │Design    │Analysis  │  ations       │  API Setup   │
 └────┬─────┘────┬─────┘──────┬────────┘──────┬───────┘
      │          │            │               │
      └──────────┴────────────┴───────┬───────┘
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ Stage 14: MLFLOW LOGGING                                                │
 │   Log params, metrics, artifacts to MLflow tracking server              │
 └─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Data Artifacts

| Stage | Output Artifact | Consumed By |
|-------|----------------|-------------|
| Data Generation | `data/customers.csv`, `data/events.csv` | Preprocessing, Feature Engineering |
| Preprocessing | `data/processed/` (train/test splits) | Feature Engineering, all Models |
| Feature Engineering | `data/features/` (30+ columns) | ML, DL, Uplift, CLV, Survival models |
| ML/DL Training | `models/*.pkl`, `models/*.pt` | Ensemble, Scoring API, Dashboard |
| Uplift Modeling | Uplift scores, 4-quadrant segments | Budget Optimizer, Recommendations |
| CLV Prediction | `results/clv_predictions.csv` | Budget Optimizer, Recommendations |
| Budget Optimization | `results/budget_results.csv`, `results/budget_whatif.csv` | Dashboard, What-if Analyzer |
| Cohort Analysis | `results/cohort_retention_curves.png` | Dashboard |
| Monitoring | `results/monitoring_report.json` | Dashboard, Alerting |

---

## 5. Module Descriptions

### 5.1 Data Layer

**Location:** `src/data/`

| Module | Class | Purpose |
|--------|-------|---------|
| `generator.py` | `CustomerDataGenerator` | Simulates realistic e-commerce customer behavior with 6 persona types, 8 event types, over a configurable time period. Generates customers with treatment/control group assignment for uplift modeling. |
| `preprocessing.py` | — | Cleans raw data, handles missing values, validates schema, applies time-based train/test split (10 months train, 2 months test). |
| `orchestrator.py` | — | Coordinates data flow between generation, preprocessing, and feature engineering stages. |

**Personas:** vip_loyal (10%), regular_loyal (25%), bargain_hunter (20%), new_customer (20%), dormant (15%), high_value_at_risk (10%)

**Event Types:** page_view, search, add_to_cart, purchase, coupon_use, review, cs_contact, remove_from_cart

---

### 5.2 Feature Engineering

**Location:** `src/features/`

| Module | Class | Purpose |
|--------|-------|---------|
| `feature_engineering.py` | `FeatureEngineer` | Computes 30+ features from raw data. Includes RFM metrics, behavioral change indicators, session quality scores, purchase cycle analysis, sequence embeddings, and temporal patterns. |
| `segmentation.py` | — | Customer segmentation using RFM-based rules and K-means clustering. Produces 6+ segments for targeted retention strategies. |

**Feature Categories:**
- **RFM:** recency, frequency, monetary scores
- **Behavioral Change:** visit_decay_rate, purchase_cycle_change, session_time_decay
- **Purchase Cycle:** avg_days_between_purchases, purchase_cycle_anomaly
- **Session Quality:** avg_session_duration, session_frequency_30d, session_bounce_rate
- **Sequence:** Event sequence embeddings for DL models
- **Temporal:** weekend_activity_ratio, time_of_day_patterns
- **Journey Stage:** Customer lifecycle position

---

### 5.3 Model Layer

**Location:** `src/models/`

#### 5.3.1 Churn Prediction (`churn_model.py`)

| Class | Architecture | Details |
|-------|-------------|---------|
| `MLChurnModel` | XGBoost / LightGBM | Gradient boosting with 5-fold cross-validation, early stopping (20 rounds) |
| `DLChurnModel` | Transformer / LSTM | PyTorch models with multi-head attention or recurrent layers; sequence window = 6 |
| `EnsembleChurnModel` | Weighted average | 0.6 × ML + 0.4 × DL for robust predictions |

#### 5.3.2 Uplift Modeling (`uplift_model.py`)

| Approach | Method |
|----------|--------|
| T-Learner | Trains separate models for treatment and control groups; CATE = E[Y|T=1] - E[Y|T=0] |
| S-Learner | Single model with treatment indicator as feature |

Produces 4-quadrant customer segmentation:
- **Persuadables** — Positive uplift (benefit from treatment)
- **Sure Things** — Low churn regardless of treatment
- **Lost Causes** — High churn regardless of treatment
- **Sleeping Dogs** — Negative uplift (harmed by treatment)

#### 5.3.3 CLV Prediction (`clv_model.py`)

Customer Lifetime Value estimation uses an ML-based 12-month value regressor with holdout validation. It integrates with churn scores and uplift scores for churn-adjusted segmentation and budget allocation.

#### 5.3.4 Survival Analysis (`survival_model.py`, `survival_analysis.py`)

- **Kaplan-Meier:** Non-parametric survival curve estimation
- **Cox Proportional Hazards:** Semi-parametric model relating features to hazard rate
- Outputs: survival functions, hazard rates, median survival times per segment

#### 5.3.5 Recommendations (`recommendations.py`)

`RecommendationEngine` generates personalized retention actions per customer based on:
- Churn risk score (from ensemble model)
- Uplift quadrant assignment
- CLV tier
- Customer segment

Produces top-K ranked actions with explanations and budget-aware filtering.

#### 5.3.6 A/B Testing (`ab_testing.py`)

Test design with power analysis, statistical testing (Chi-square, Z-test), confidence intervals, and minimum sample size calculations.

#### 5.3.7 Explainability (`shap_explainer.py`)

`ShapExplainer` computes SHAP values for global and local feature importance, producing summary plots for model interpretability.

#### 5.3.8 Experiment Tracking (`mlflow_tracking.py`)

| Class | Purpose |
|-------|---------|
| `MLflowTracker` | Logs parameters, metrics, and artifacts to MLflow experiments |
| `ModelRegistry` | Registers model versions, manages staging/production transitions |

---

### 5.4 Optimization Layer

**Location:** `src/optimization/`, `src/models/budget_optimizer.py`, `src/models/whatif_analysis.py`

**Budget Optimization Problem:**

```
maximize:   Σ (uplift_i × CLV_i × action_i)

subject to:
  Σ (cost_channel × action_i) ≤ Total Budget (50M KRW default)
  Per-channel budget limits
  action_i ∈ {0, 1} per customer per channel
```

**Marketing Channels:**

| Channel | Cost/Action (KRW) | ROI Multiplier |
|---------|-------------------|----------------|
| Email | 1,000 | 1.0 |
| SMS | 500 | 0.8 |
| Push Notification | 200 | 0.6 |
| Coupon | 5,000 | 1.5 |
| Call Center | 15,000 | 2.0 |

**What-If Analysis:** Evaluates budget scenarios at 50%, 100%, and 200% of base allocation to inform budget decisions.

---

### 5.5 Analysis Layer

**Location:** `src/analysis/`

| Module | Class | Purpose |
|--------|-------|---------|
| `ab_testing.py` | `ABTestAnalyzer` | Power analysis for experiment design, statistical significance testing (Chi-square, Z-test), confidence interval computation, minimum sample size calculation |
| `cohort_analysis.py` | `CohortAnalyzer` | Monthly/weekly/behavioral cohort construction, retention curves, cumulative revenue tracking, churn rate comparison across cohorts |

**Cohort Metrics:** Retention rates at M1/M3/M6/M12, average order value trends, revenue per cohort, inter-cohort churn rate comparison.

---

### 5.6 Monitoring Layer

**Location:** `src/monitoring/`

| Module | Class | Method | Alert Thresholds |
|--------|-------|--------|-----------------|
| `drift_detection.py` | `DriftDetector` | Population Stability Index (PSI) | GREEN < 0.10, YELLOW < 0.25, RED ≥ 0.25 |
| `ks_drift.py` | `KSDriftDetector` | Kolmogorov-Smirnov test | GREEN: p > 0.05, YELLOW: p > 0.01, RED: p ≤ 0.01 |
| `monitoring_service.py` | `MonitoringService` | Orchestrator | Combines PSI + KS, logs to MLflow |

Monitoring runs automatically after model inference and detects feature-level data drift between training and scoring distributions.

---

### 5.7 Streaming Layer

**Location:** `src/streaming/`

Provides real-time scoring via Redis Streams for low-latency inference.

```
┌────────────┐     scoring_requests     ┌──────────────┐
│   Redis    │ ◄──────────────────────── │   Producer   │
│   Stream   │                           │ (publishes   │
│            │ ────────────────────────► │  requests)   │
│            │     scoring_responses     └──────────────┘
│            │ ────────────────────────►
│            │                           ┌──────────────┐
│            │ ◄──────────────────────── │   Consumer   │
└────────────┘   consumer group          │ (scores &    │
                                         │  responds)   │
                                         └──────────────┘
```

- **Stream:** `scoring_requests` (max 10K entries)
- **Consumer Group:** `scoring_consumers`
- **Batch Size:** 10 requests per poll
- **Cache TTL:** 3600 seconds

---

### 5.8 Pipeline Orchestration

**Location:** `src/pipeline/`

| Module | Class | Purpose |
|--------|-------|---------|
| `runner.py` | `PipelineRunner` | Executes the 16-stage pipeline with dependency management, error handling, and checkpoint support |
| `pipeline_state.py` | `PipelineState` | Persists pipeline state (pending/completed/failed) to enable resume after failure |

**14 Pipeline Stages:**

| # | Stage | Depends On |
|---|-------|-----------|
| 1 | data_generation | — |
| 2 | preprocessing | 1 |
| 3 | feature_engineering | 2 |
| 4 | ml_model_training | 3 |
| 5 | dl_model_training | 3 |
| 6 | ensemble_creation | 4, 5 |
| 7 | uplift_modeling | 3, 6 |
| 8 | clv_prediction | 3, 6 |
| 9 | budget_optimization | 7, 8 |
| 10 | ab_testing | 6 |
| 11 | survival_analysis | 3 |
| 12 | recommendations | 7, 8, 9 |
| 13 | scoring_api_setup | 6 |
| 14 | mlflow_logging | 6 |

---

### 5.9 Dashboard Layer

**Location:** `src/dashboard/`

Multi-page Streamlit application served on port 8501 with the following views:

| View | Description |
|------|-------------|
| **Overview** | KPI cards, churn distribution, key metrics summary |
| **Model Performance** | ML vs DL vs Ensemble comparison, ROC/PR curves |
| **Customer Segments** | RFM-based segments, uplift quadrant visualization |
| **Budget Optimizer** | LP results, channel allocation, what-if scenarios |
| **A/B Testing** | Power analysis, test results, confidence intervals |
| **Survival Analysis** | Kaplan-Meier curves, median survival by segment |
| **Recommendations** | Per-customer retention actions with explanations |
| **CLV Analysis** | Value distribution, top customers, churn-adjusted CLV |
| **Monitoring** | Feature-level drift alerts (PSI/KS), trend charts |
| **System Health** | Pipeline status, model freshness, system diagnostics |

---

## 6. Infrastructure & Deployment

### 6.1 Docker Compose Services

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────┐  ┌────────────┐  │
│  │   pipeline    │  │  dashboard   │  │redis │  │   mlflow   │  │
│  │ Python 3.10   │  │ Streamlit    │  │  7   │  │  Server    │  │
│  │               │  │ :8501        │  │:6379 │  │  :5000     │  │
│  │ ML/DL train   │  │ Interactive  │  │      │  │  SQLite    │  │
│  │ Full pipeline │  │ 10+ views    │  │Stream│  │  backend   │  │
│  └──────┬────────┘  └──────┬───────┘  └──┬───┘  └─────┬──────┘  │
│         │                  │             │             │          │
│         └──────────────────┴──────┬──────┴─────────────┘          │
│                                   │                               │
│                          Shared Volumes                           │
│                   config/ data/ models/ results/                  │
└─────────────────────────────────────────────────────────────────┘
```

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| `pipeline` | Dockerfile.pipeline | — | ML/DL training, data generation, full pipeline execution |
| `dashboard` | Dockerfile.dashboard | 8501 | Streamlit interactive dashboard |
| `redis` | redis:7-alpine | 6379 | Real-time scoring streams, feature cache |
| `mlflow` | Dockerfile.mlflow | 5000 | Experiment tracking, model registry |

### 6.2 Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./config/` | `/app/config` | YAML configuration files |
| `./data/` | `/app/data` | Raw and processed data |
| `./models/` | `/app/models` | Trained model artifacts |
| `./results/` | `/app/results` | Output results and reports |
| `./mlruns/` | `/mlflow/artifacts` | MLflow experiment artifacts |

---

## 7. Configuration Management

All configurable parameters are centralized in `config/simulator_config.yaml` (531 lines). Key sections:

| Section | Parameters |
|---------|-----------|
| `simulation` | num_customers, simulation_months, random_seed, start_date |
| `churn_definition` | no_purchase_days (30), no_login_days (60), operator (OR) |
| `treatment` | treatment_ratio (0.50), min_group_size (10,000) |
| `pipeline` | train_months (10), test_months (2) |
| `personas` | 6 personas with behavioral parameters and proportions |
| `ml_model` | n_folds (5), early_stopping_rounds (20) |
| `dl_model` | architecture, hidden_size (64), num_layers (2), num_attention_heads (4) |
| `optimization` | total_budget (50M KRW), channel costs and limits |
| `monitoring` | PSI/KS thresholds, drift detection toggles |
| `segmentation` | method (rfm_behavioral), n_segments (8+) |
| `small_mode` | Reduced settings for development (5K customers, 6 months) |

---

## 8. Testing Architecture

The project follows a TDD approach with 54+ test files in `tests/`.

### Test Categories

| Category | Test Files | Coverage |
|----------|-----------|----------|
| **Data** | test_data_generator, test_preprocessing, test_orchestrator | Data generation, cleaning, orchestration |
| **Features** | test_feature_engineering, test_segmentation, test_sequence_dataset | 30+ features, segments, sequences |
| **Core Models** | test_churn_model, test_dl_trainer, test_shap_explainer | ML/DL training, SHAP explanations |
| **Advanced Models** | test_uplift_model, test_clv, test_survival_analysis, test_recommendations | Uplift, CLV, survival, recommendations |
| **Optimization** | test_budget_optimizer, test_budget_lp_solver, test_whatif_analysis | LP solver, scenarios, cost config |
| **Analysis** | test_ab_testing, test_statistical_testing, test_cohort_analysis, test_experiment_manager | A/B tests, cohorts, power analysis |
| **Monitoring** | test_drift_detection, test_ks_drift, test_monitoring_service | PSI, KS-test, alerting |
| **Infrastructure** | test_pipeline_runner, test_pipeline_state, test_mlflow_tracking, test_model_registry | Pipeline, MLflow, checkpoints |
| **Serving** | test_dashboard, test_dashboard_helpers, test_redis_streaming, test_scoring_api | Dashboard, streaming, API |
| **Integration** | test_integration, test_docker_setup | End-to-end, Docker validation |
| **CLI** | test_main_cli | All 14 modes, argument parsing |

### Test Fixtures

Shared fixtures in `tests/conftest.py` provide:
- Configuration loaded from YAML
- Sample customer/event DataFrames
- Treatment/control group data for uplift tests
- Streamlit mocks for dashboard tests
- Temporary directories for artifact persistence tests

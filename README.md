# E-Commerce Customer Churn Prediction & Retention Optimization System

> End-to-end ML/DL pipeline for predicting customer churn, identifying uplift segments, optimizing retention budgets, and delivering actionable insights via a real-time dashboard.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org)

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [Docker Compose Usage](#docker-compose-usage)
- [Pipeline Modes](#pipeline-modes)
- [Configuration](#configuration)
- [Outputs & Deliverables](#outputs--deliverables)
- [Testing](#testing)
- [Documentation](#documentation)
- [License](#license)

---

## Project Overview

Customer churn is one of the most critical challenges in e-commerce. Acquiring a new customer costs **5x more** than retaining an existing one, and reducing churn by just **5%** can increase revenue by **25% or more**.

This system goes beyond simple churn prediction. It builds a complete retention optimization pipeline that:

1. **Simulates** realistic customer behavior across 6 personas over 12 months
2. **Engineers** 30+ features including RFM, behavioral changes, and sequence embeddings
3. **Predicts** churn using ensemble ML (XGBoost/LightGBM) and DL (LSTM/Transformer) models
4. **Identifies** which customers respond to marketing via Uplift Modeling
5. **Estimates** Customer Lifetime Value (CLV) for prioritization
6. **Optimizes** marketing budget allocation under constraints to maximize ROI
7. **Validates** strategies through A/B testing with statistical rigor
8. **Monitors** model performance and data drift in production
9. **Visualizes** everything through an interactive Streamlit dashboard

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Environment                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Pipeline Container (pipeline)                  │   │
│  │                                                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │   │
│  │  │Simulator │→ │ Feature  │→ │  Model   │→ │   Uplift     │    │   │
│  │  │(20K cust)│  │Engineer  │  │Training  │  │  Modeling     │    │   │
│  │  │          │  │(30+ feat)│  │(ML + DL) │  │(T/S-Learner) │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │   │
│  │       │                            │               │            │   │
│  │       ▼                            ▼               ▼            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │   │
│  │  │ Cohort   │  │   CLV    │  │  Budget  │  │   A/B Test   │    │   │
│  │  │ Analysis │  │Prediction│  │Optimizer │  │   Analysis   │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │   │
│  │       │              │             │               │            │   │
│  │       ▼              ▼             ▼               ▼            │   │
│  │  ┌──────────────────────────────────────────────────────┐       │   │
│  │  │         Pipeline State (pipeline_state.json)         │       │   │
│  │  │    Checkpoint: completed / failed / pending           │       │   │
│  │  └──────────────────────────────────────────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐      │
│  │   Dashboard     │  │     Redis      │  │      MLflow          │      │
│  │  (Streamlit)    │  │  (Real-time    │  │   (Experiment        │      │
│  │  localhost:8501 │  │   Scoring &    │  │    Tracking)         │      │
│  │                 │  │   Feature      │  │   localhost:5000     │      │
│  │  - Churn Dist.  │  │   Store)       │  │                      │      │
│  │  - Cohort Curves│  │               │  │   - Metrics           │      │
│  │  - Uplift 4-Quad│  │  localhost:    │  │   - Parameters        │      │
│  │  - CLV Analysis │  │    6379       │  │   - Artifacts          │      │
│  │  - Budget Optim.│  │               │  │   - Model Registry     │      │
│  │  - A/B Results  │  │               │  │                      │      │
│  │  - Segments     │  │               │  │   (SQLite backend)    │      │
│  │  - Monitoring   │  │               │  │                      │      │
│  └────────────────┘  └────────────────┘  └──────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Shared Volumes                                 │   │
│  │  ./data/     ./models/     ./results/     ./mlruns/     ./config/ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
config/*.yaml ──→ Simulator ──→ data/raw/ ──→ Feature Engineering ──→ data/features/
                                                      │
                                    ┌─────────────────┼──────────────────┐
                                    ▼                 ▼                  ▼
                              ML Models          DL Models         Uplift Models
                              (XGBoost,          (LSTM/            (T-Learner,
                               LightGBM)         Transformer)      S-Learner)
                                    │                 │                  │
                                    ▼                 ▼                  ▼
                              Ensemble (0.6 ML + 0.4 DL)          Uplift Scores
                                    │                                    │
                                    ▼                                    ▼
                              CLV Prediction ──→ 6-Segment Classification
                                                         │
                                                         ▼
                                              Budget Optimization
                                              (LP: max Σ Uplift×CLV×Action)
                                                         │
                                    ┌────────────────────┼────────────────┐
                                    ▼                    ▼                ▼
                              A/B Test Design    Retention Strategy   Dashboard
                              & Analysis         & Recommendations    (Streamlit)
```

---

## Features

### Core Features (13 Mandatory)

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Customer Behavior Simulator** | Generates 20K+ customers with 6 personas, 8+ event types, 12 months of behavioral data, treatment/control groups, configurable churn definition |
| 2 | **Cohort & Journey Analysis** | Monthly cohort retention curves (M1/M3/M6/M12), churn pattern extraction, customer journey funnel analysis |
| 3 | **Feature Engineering** | 30+ features: RFM, behavioral change rates, purchase cycle anomaly, session quality, sequence embeddings, temporal patterns |
| 4 | **ML Churn Prediction** | XGBoost + LightGBM with class imbalance handling, 5-fold CV, SHAP interpretation, threshold optimization, hyperparameter tuning |
| 5 | **DL Churn Prediction** | LSTM/Transformer on behavioral sequences with early stopping, ensemble with ML models (0.6 ML + 0.4 DL) |
| 6 | **Uplift Modeling** | T-Learner & S-Learner, CATE estimation, 4-quadrant segmentation (Persuadables/Sure Things/Lost Causes/Sleeping Dogs), Qini curve |
| 7 | **CLV Prediction** | BG/NBD + Gamma-Gamma or ML-based, 12-month forecast, top-20% high-value identification |
| 8 | **Customer Segmentation** | 6+ segments combining churn probability, uplift score, CLV; priority scoring (Uplift × CLV) |
| 9 | **Budget Optimization** | Linear programming under budget constraints (default 50M KRW), what-if analysis (50%/100%/200%), ROI estimation |
| 10 | **A/B Test Design & Analysis** | Power analysis, Chi-square/Z-test, 95% CI, p-value reporting, result interpretation |
| 11 | **Integrated Dashboard** | Streamlit on port 8501 with churn distribution, cohort curves, uplift segments, CLV analysis, budget simulator, A/B results, monitoring |
| 12 | **Model Monitoring** | PSI/KS-test for data drift detection, performance tracking over time, alert generation |
| 13 | **Documentation & Code Quality** | Architecture diagrams, feature dictionary, model report, retention strategy, PEP8 compliance, docstrings |

### Bonus Features (4)

| # | Feature | Description |
|---|---------|-------------|
| B1 | **Real-time Scoring** | Redis-backed streaming inference for individual customer scoring |
| B2 | **Survival Analysis** | Kaplan-Meier / Cox PH models for time-to-churn estimation |
| B3 | **Personalized Recommendations** | Per-segment tailored retention action suggestions |
| B4 | **MLflow Tracking** | Full experiment tracking with metrics, parameters, artifacts, and model registry |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **ML Models** | XGBoost, LightGBM, scikit-learn, SHAP, Optuna |
| **DL Models** | PyTorch 2.0+ (LSTM, Transformer) |
| **Uplift** | causalml or custom T-Learner/S-Learner |
| **CLV** | lifetimes (BG/NBD + Gamma-Gamma) |
| **Optimization** | scipy.optimize / PuLP (linear programming) |
| **Statistics** | scipy.stats, statsmodels (A/B testing, survival analysis) |
| **Dashboard** | Streamlit |
| **Experiment Tracking** | MLflow (SQLite + local artifacts) |
| **Feature Store** | File-based (Parquet/CSV) + Redis for real-time |
| **Containerization** | Docker, Docker Compose |
| **Caching/Streaming** | Redis |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Testing** | pytest, pytest-cov |
| **Configuration** | YAML (PyYAML) |

---

## Project Structure

```
capstone/
├── config/                          # All YAML configuration files
│   ├── base_config.yaml             # Global settings (seed, paths, churn definition)
│   ├── simulator_config.yaml        # Persona definitions & simulation parameters
│   ├── feature_config.yaml          # Feature engineering settings
│   ├── model_config.yaml            # ML/DL model hyperparameters
│   ├── uplift_config.yaml           # Uplift modeling settings
│   ├── optimization_config.yaml     # Budget optimization parameters
│   └── dashboard_config.yaml        # Dashboard layout settings
│
├── data/                            # Data directory (generated)
│   ├── raw/                         # Raw simulated event logs
│   └── features/                    # Engineered feature store (file-based)
│
├── docs/                            # Documentation deliverables
│   ├── feature_dictionary.md        # 30+ feature definitions & business meaning
│   ├── model_report.md              # ML/DL comparison & model selection rationale
│   ├── retention_strategy.md        # 6-segment retention strategies
│   ├── ab_test_report.md            # A/B test power analysis & results
│   └── uplift_analysis.md           # Qini curve & 4-quadrant interpretation
│
├── models/                          # Trained model artifacts
│
├── results/                         # All output artifacts
│   ├── shap_summary.png             # SHAP global feature importance plot
│   ├── clv_predictions.csv          # Customer-level CLV with Top-20% flag
│   ├── segments_6plus.csv           # 6-segment classification with priority scores
│   ├── monitoring_report.json       # PSI/KS drift metrics & threshold alerts
│   └── ...                          # Cohort curves, uplift plots, etc.
│
├── src/                             # Source code
│   ├── main.py                      # CLI entry point (--mode train|uplift|optimize)
│   ├── data/
│   │   ├── simulator.py             # Customer behavior simulator
│   │   └── preprocessor.py          # Data cleaning & preparation
│   ├── features/
│   │   ├── engineer.py              # Feature engineering pipeline
│   │   └── store.py                 # Feature store (file + Redis)
│   ├── models/
│   │   ├── ml_models.py             # XGBoost, LightGBM training & evaluation
│   │   ├── dl_models.py             # LSTM/Transformer PyTorch models
│   │   └── ensemble.py              # Weighted ensemble (0.6 ML + 0.4 DL)
│   ├── uplift/
│   │   ├── uplift_model.py          # T-Learner, S-Learner implementation
│   │   └── segmentation.py          # 4-quadrant & 6-segment classification
│   ├── clv/
│   │   └── clv_predictor.py         # CLV estimation (BG/NBD + Gamma-Gamma)
│   ├── optimization/
│   │   └── budget_optimizer.py      # LP-based budget allocation
│   ├── ab_testing/
│   │   └── ab_analyzer.py           # Power analysis, significance testing
│   ├── monitoring/
│   │   └── drift_detector.py        # PSI/KS data drift detection
│   ├── analysis/
│   │   └── cohort_analysis.py       # Cohort retention & journey analysis
│   ├── streaming/
│   │   └── realtime_scorer.py       # Redis-backed real-time scoring
│   ├── survival/
│   │   └── survival_analysis.py     # Kaplan-Meier & Cox PH models
│   ├── recommendations/
│   │   └── recommender.py           # Personalized retention recommendations
│   └── utils/
│       ├── config_loader.py         # YAML config loading utilities
│       ├── logger.py                # Logging setup
│       └── pipeline_state.py        # Checkpoint management (pipeline_state.json)
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard application
│
├── tests/                           # TDD test suite
│   ├── test_simulator.py
│   ├── test_features.py
│   ├── test_ml_models.py
│   ├── test_dl_models.py
│   ├── test_uplift.py
│   ├── test_clv.py
│   ├── test_optimization.py
│   ├── test_ab_testing.py
│   ├── test_monitoring.py
│   ├── test_pipeline.py
│   └── ...
│
├── mlruns/                          # MLflow tracking data (SQLite + artifacts)
├── pipeline_state.json              # Pipeline checkpoint state
├── docker-compose.yml               # 4-container orchestration
├── Dockerfile.pipeline              # Pipeline container image
├── Dockerfile.dashboard             # Dashboard container image
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── require.md                       # Original requirements specification
```

---

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/<your-username>/capstone.git
cd capstone

# Start all services (pipeline, dashboard, redis, mlflow)
docker-compose up --build

# Access the dashboard
open http://localhost:8501

# Access MLflow UI
open http://localhost:5000
```

### Option 2: Local Development

```bash
# Clone and enter directory
git clone https://github.com/<your-username>/capstone.git
cd capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/main.py --mode train
python src/main.py --mode uplift
python src/main.py --mode optimize --budget 50000000

# Launch dashboard
streamlit run dashboard/app.py --server.port 8501
```

---

## Installation & Setup

### Prerequisites

- **Docker** >= 20.10 and **Docker Compose** >= 2.0 (for containerized setup)
- **Python** >= 3.10 (for local development)
- **Redis** (automatically provisioned via Docker Compose, or install locally)
- No GPU required — all models run on CPU

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/capstone.git
   cd capstone
   ```

2. **Configure parameters** (optional — sensible defaults are provided):
   ```bash
   # Edit configuration files in config/ directory
   vim config/base_config.yaml        # Seed, paths, churn definition
   vim config/simulator_config.yaml   # Customer count, persona weights
   vim config/model_config.yaml       # Hyperparameters, ensemble weights
   ```

3. **Verify setup:**
   ```bash
   # Run tests
   pytest tests/ -v

   # Check configuration
   python -c "from src.utils.config_loader import load_config; print(load_config())"
   ```

---

## Docker Compose Usage

### Container Architecture

| Container | Purpose | Port | Image Base |
|-----------|---------|------|------------|
| `pipeline` | ML/DL training, data generation, optimization | — | Python 3.10-slim |
| `dashboard` | Streamlit interactive dashboard | 8501 | Python 3.10-slim |
| `redis` | Real-time feature store & streaming scoring | 6379 | redis:7-alpine |
| `mlflow` | Experiment tracking server | 5000 | Python 3.10-slim |

### Commands

```bash
# Build and start all containers
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Run only the pipeline
docker-compose up pipeline

# Run specific pipeline mode
docker-compose run pipeline python src/main.py --mode train
docker-compose run pipeline python src/main.py --mode uplift
docker-compose run pipeline python src/main.py --mode optimize --budget 50000000

# View logs
docker-compose logs -f dashboard
docker-compose logs -f pipeline

# Stop all containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Shared Volumes

All containers share these mounted volumes for data persistence:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data/` | `/app/data/` | Raw & processed data |
| `./models/` | `/app/models/` | Trained model files |
| `./results/` | `/app/results/` | Output artifacts |
| `./mlruns/` | `/app/mlruns/` | MLflow tracking data |
| `./config/` | `/app/config/` | YAML configuration |

---

## Pipeline Modes

### Train Mode
```bash
python src/main.py --mode train
```
Executes the full training pipeline:
1. Generate simulated customer data (or use existing)
2. Perform cohort and journey analysis
3. Engineer features (30+ features)
4. Train ML models (XGBoost, LightGBM) with 5-fold CV
5. Train DL model (LSTM/Transformer) with early stopping
6. Create ensemble predictions (0.6 ML + 0.4 DL)
7. Generate SHAP explanations
8. Predict CLV
9. Run model monitoring & drift detection
10. Track experiments in MLflow

**Outputs:** `models/`, `results/shap_summary.png`, `results/monitoring_report.json`

### Uplift Mode
```bash
python src/main.py --mode uplift
```
Runs uplift modeling and segmentation:
1. Train T-Learner and S-Learner models
2. Calculate customer-level Uplift Scores (CATE)
3. Classify into 4 quadrants (Persuadables, Sure Things, Lost Causes, Sleeping Dogs)
4. Create 6+ segments combining churn probability, uplift, and CLV
5. Generate Qini curves and segment visualizations

**Outputs:** `results/segments_6plus.csv`, uplift plots in `results/`

### Optimize Mode
```bash
python src/main.py --mode optimize --budget 50000000
```
Runs budget optimization and A/B test analysis:
1. Formulate LP: maximize Σ(Uplift_i × CLV_i × Action_i) subject to budget constraint
2. Allocate budget across segments
3. Run what-if analysis (50%, 100%, 200% budget scenarios)
4. Perform A/B test design with power analysis
5. Calculate statistical significance and ROI estimates

**Outputs:** Budget allocation reports, `docs/ab_test_report.md`

### Pipeline Checkpointing

The pipeline tracks execution state in `pipeline_state.json`:

```json
{
  "simulation": "completed",
  "feature_engineering": "completed",
  "ml_training": "completed",
  "dl_training": "pending",
  "uplift_modeling": "pending",
  "optimization": "pending",
  "timestamp": "2026-03-21T10:00:00"
}
```

On restart, completed steps are skipped — only `pending` or `failed` steps are re-executed.

---

## Configuration

All parameters are managed via YAML files in the `config/` directory.

### Key Configuration Parameters

**`config/base_config.yaml`** — Global settings:
```yaml
random_seed: 42
churn_definition:
  no_purchase_days: 30      # Days without purchase to flag churn
  no_login_days: 60         # Days without login to flag churn
paths:
  data_raw: data/raw/
  data_features: data/features/
  models: models/
  results: results/
```

**`config/simulator_config.yaml`** — Simulation parameters:
```yaml
num_customers: 20000         # Total customers (small mode: 5000)
simulation_months: 12        # Duration (small mode: 6)
personas:
  - name: VIP_loyal          # VIP충성고객
  - name: regular_loyal      # 일반충성고객
  - name: price_sensitive    # 가격민감형
  - name: explorer           # 탐색형
  - name: churning           # 이탈진행형
  - name: new_user           # 신규가입자
treatment_ratio: 0.5         # 50% treatment, 50% control
target_churn_rate: [0.15, 0.25]  # 15-25% churn rate range
```

**`config/model_config.yaml`** — Model parameters:
```yaml
train_test_split:
  method: time_based
  train_months: 10
  test_months: 2
ensemble:
  ml_weight: 0.6
  dl_weight: 0.4
optimization:
  default_budget: 50000000   # 50M KRW
```

### Reproducibility

Setting the same `random_seed` in `config/base_config.yaml` guarantees identical results across runs (same data generation, train/test splits, model initialization, etc.).

---

## Outputs & Deliverables

| Deliverable | Path | Verification |
|------------|------|-------------|
| Feature Dictionary | `docs/feature_dictionary.md` | 30+ feature definitions |
| Model Report | `docs/model_report.md` | ML/DL comparison included |
| Retention Strategy | `docs/retention_strategy.md` | 6-segment strategies |
| CLV Predictions | `results/clv_predictions.csv` | Per-customer CLV + Top-20% |
| 6-Segment Results | `results/segments_6plus.csv` | Priority score included |
| Monitoring Report | `results/monitoring_report.json` | PSI/KS + threshold alerts |
| A/B Test Report | `docs/ab_test_report.md` | Power analysis + p-value |
| Uplift Analysis | `docs/uplift_analysis.md` | Qini curve + 4-quadrant |
| SHAP Summary Plot | `results/shap_summary.png` | Top 10 feature importance |

---

## Testing

This project follows a **TDD (Test-Driven Development)** approach.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_simulator.py -v
pytest tests/test_ml_models.py -v

# Run tests in Docker
docker-compose run pipeline pytest tests/ -v
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Feature Dictionary](docs/feature_dictionary.md) | Complete definitions of 30+ engineered features with business rationale |
| [Model Report](docs/model_report.md) | ML/DL training process, performance comparison, and model selection |
| [Retention Strategy](docs/retention_strategy.md) | Segment-specific retention strategies with expected costs and effects |
| [A/B Test Report](docs/ab_test_report.md) | Statistical test design, power analysis, and result interpretation |
| [Uplift Analysis](docs/uplift_analysis.md) | Uplift modeling methodology, Qini curves, and segment characteristics |

---

## License

This project is developed as a capstone design project. All rights reserved.

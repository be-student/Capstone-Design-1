# E-Commerce Customer Churn Prediction & Retention Optimization System

> Documentation Hub for the end-to-end ML/DL customer churn prediction and retention optimization platform.

---

## Project Overview

Customer churn is one of the most critical challenges in e-commerce. Acquiring a new customer costs **5x more** than retaining an existing one, and reducing churn by just **5%** can increase revenue by **25% or more**.

This system delivers a complete retention optimization pipeline that goes far beyond simple churn prediction:

- **Simulate** realistic customer behavior across 6 personas over 12 months (20,000+ customers)
- **Engineer** 30+ features including RFM metrics, behavioral change indicators, session quality scores, and sequence embeddings
- **Predict** churn using an ensemble of ML (XGBoost / LightGBM) and DL (Transformer / LSTM) models with 5-fold cross-validation
- **Model uplift** to identify which customers actually respond to marketing interventions (T-Learner & S-Learner)
- **Estimate CLV** per customer using BG/NBD-inspired gradient boosting for 12-month forward value prediction
- **Optimize budgets** via linear programming under real-world constraints across 5 marketing channels
- **Design & analyze A/B tests** with power analysis, Chi-square/Z-tests, and 95% confidence intervals
- **Analyze survival** patterns with Kaplan-Meier and Cox Proportional Hazards models
- **Generate recommendations** for personalized retention actions ranked by expected impact
- **Monitor** model performance and data drift (PSI / KS-test) with automated alerting
- **Visualize** all insights through an interactive Streamlit dashboard with 10+ views

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **13 Mandatory Features** | Simulation, cohort analysis, feature engineering, ML/DL churn prediction, uplift modeling, CLV, segmentation, budget optimization, A/B testing, dashboard, monitoring, documentation |
| **4 Bonus Features** | Real-time scoring (Redis), survival analysis, personalized recommendations, MLflow experiment tracking |
| **14-Stage Pipeline** | Fully orchestrated with checkpoint/resume support |
| **4 Docker Services** | Pipeline, Dashboard, MLflow, Redis -- all containerized |

---

## System Architecture

The platform follows a layered architecture with clear separation of concerns. Each layer communicates through well-defined data artifacts (CSV, Parquet, JSON, pickle).

### Architecture Layers

```
+------------------------------------------------------------------+
|                    PRESENTATION LAYER                              |
|     Streamlit Dashboard (:8501)  |  MLflow UI (:5000)             |
|     10+ interactive views        |  Experiment tracking            |
+----------------------------------+-------------------------------+
                    |                              |
+------------------------------------------------------------------+
|                    SERVING LAYER                                   |
|     Redis Streams (:6379) -- Real-time scoring & caching          |
|     Scoring API -- Batch & online inference                       |
+------------------------------------------------------------------+
                    |
+------------------------------------------------------------------+
|                    ANALYSIS LAYER                                  |
|     A/B Testing     |  Cohort Analysis   |  What-If Scenarios     |
|     Power analysis  |  Retention curves  |  Budget scenarios      |
+------------------------------------------------------------------+
                    |
+------------------------------------------------------------------+
|                    OPTIMIZATION LAYER                              |
|     Budget Optimizer (LP)  |  Recommendation Engine               |
|     maximize SUM(uplift_i * CLV_i * action_i)                     |
|     subject to: total_cost <= Budget                              |
+------------------------------------------------------------------+
                    |
+------------------------------------------------------------------+
|                    MODEL LAYER                                     |
|  Churn Prediction  | Uplift Modeling | CLV Prediction | Survival  |
|  XGBoost/LightGBM  | T/S-Learner    | GBM (BG/NBD)  | KM / Cox  |
|  Transformer/LSTM  | 4-quadrant seg | 12-month fwd   | Hazard    |
|  Ensemble (0.6/0.4)| CATE scores    | Top-20% ID     | Curves    |
+------------------------------------------------------------------+
                    |
+------------------------------------------------------------------+
|                    FEATURE LAYER                                   |
|  RFM Metrics | Behavioral Change | Session Quality | Sequences   |
|  30+ engineered features with temporal & journey-stage indicators |
+------------------------------------------------------------------+
                    |
+------------------------------------------------------------------+
|                    DATA LAYER                                      |
|  Customer Simulator (6 personas, 8 event types, 12 months)       |
|  Preprocessing & Validation | Train/Test Split (10mo/2mo)        |
+------------------------------------------------------------------+
                    |
+------------------------------------------------------------------+
|                    INFRASTRUCTURE                                  |
|  Docker Compose | Config (YAML) | MLflow Tracking | Redis Cache  |
|  Pipeline Orchestration (14 stages with checkpoints)              |
+------------------------------------------------------------------+
```

### Pipeline Flow

The CLI entrypoint (`src/main.py`) drives a 14-stage pipeline with dependency management:

```
Data Generation --> Preprocessing --> Feature Engineering
                                           |
                              +------------+------------+
                              |                         |
                        ML Training              DL Training
                        (XGBoost/LGBM)           (Transformer/LSTM)
                              |                         |
                              +------------+------------+
                                           |
                                    Ensemble Creation
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
              Uplift Modeling        CLV Prediction        A/B Test Design
              (T/S-Learner)         (Gradient Boost)
                    |                      |
                    +----------+-----------+
                               |
                     Budget Optimization (LP)
                               |
                    +----------+-----------+----------+
                    |                      |          |
              Recommendations      Survival      Scoring API
                                   Analysis       + MLflow
```

### Docker Compose Services

```
+---------------------------------------------------------------+
|                   Docker Compose Network                        |
|                                                                 |
|  +-------------+  +-------------+  +--------+  +------------+ |
|  |  pipeline   |  |  dashboard  |  | redis  |  |   mlflow   | |
|  | Python 3.10 |  | Streamlit   |  |  7     |  |  Server    | |
|  |             |  | :8501       |  | :6379  |  |  :5000     | |
|  | ML/DL train |  | 10+ views   |  | Stream |  |  SQLite    | |
|  +------+------+  +------+------+  +---+----+  +-----+------+ |
|         |                |             |              |         |
|         +----------------+------+------+--------------+         |
|                                 |                               |
|                        Shared Volumes                           |
|                 config/ data/ models/ results/                   |
+---------------------------------------------------------------+
```

| Service | Port | Purpose |
|---------|------|---------|
| `pipeline` | -- | Runs the full ML/DL training pipeline (all 14 stages) |
| `dashboard` | 8501 | Interactive Streamlit dashboard with 10+ views |
| `redis` | 6379 | Real-time scoring streams and feature caching |
| `mlflow` | 5000 | Experiment tracking, model registry, artifact storage |

---

## Project Structure

```
capstone/
+-- src/                          # Source code
|   +-- main.py                   # CLI entrypoint (14 modes)
|   +-- __main__.py               # python -m src support
|   +-- data/                     # Data generation, preprocessing
|   +-- features/                 # Feature engineering, segmentation
|   +-- models/                   # ML/DL models, uplift, CLV, survival, recommendations
|   +-- optimization/             # Budget optimizer (LP)
|   +-- analysis/                 # A/B testing, cohort analysis
|   +-- monitoring/               # Drift detection (PSI, KS-test)
|   +-- streaming/                # Redis producer/consumer
|   +-- pipeline/                 # 14-stage pipeline orchestration
|   +-- dashboard/                # Streamlit app (10+ views)
|
+-- config/                       # YAML configuration files
|   +-- simulator_config.yaml     # All parameters (personas, models, budget, etc.)
|
+-- tests/                        # 54+ test files (TDD approach)
+-- docs/                         # Documentation
|   +-- README.md                 # This file
|   +-- architecture.md           # Detailed architecture reference
|   +-- models.md                 # Model training & performance
|   +-- api.md                    # API reference
|   +-- deployment.md             # Deployment guide
|   +-- usage.md                  # Usage guide
|
+-- data/                         # Raw & processed data
+-- models/                       # Trained model artifacts
+-- results/                      # Output CSV, JSON, PNG artifacts
+-- scripts/                      # Shell scripts for Docker entrypoints
+-- docker-compose.yml            # 4-service orchestration
+-- Dockerfile.pipeline           # Pipeline container
+-- Dockerfile.dashboard          # Dashboard container
+-- Dockerfile.mlflow             # MLflow container
+-- requirements.txt              # Python dependencies
```

---

## Quick Start

### Option 1: Docker Compose (Recommended)

The fastest way to run the entire system end-to-end:

```bash
# Clone the repository
git clone <repository-url>
cd capstone

# Build and start all services
docker-compose up --build

# Access the services:
#   Dashboard:  http://localhost:8501
#   MLflow UI:  http://localhost:5001
```

The pipeline container will automatically:
1. Generate simulated customer data (20K customers, 12 months)
2. Engineer 30+ features
3. Train ML/DL models with 5-fold cross-validation
4. Run uplift modeling, CLV prediction, and budget optimization
5. Output results to `results/` and `models/`

Once the pipeline completes, the dashboard container starts serving on port 8501.

To stop all services:

```bash
docker-compose down
```

### Option 2: Local Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Run the Full Pipeline

```bash
# Full pipeline (all 14 stages)
python -m src.main --mode all

# Small mode for development (5K customers, 6 months)
python -m src.main --mode all --small
```

#### Run Individual Modes

```bash
# Data simulation
python -m src.main --mode simulate

# Model training (ML + DL + Ensemble)
python -m src.main --mode train

# Uplift modeling & 4-quadrant segmentation
python -m src.main --mode uplift

# CLV prediction
python -m src.main --mode clv

# Budget optimization with specific budget (KRW)
python -m src.main --mode optimize --budget 50000000

# A/B test design & analysis
python -m src.main --mode ab_test

# Survival analysis (Kaplan-Meier + Cox PH)
python -m src.main --mode survival

# Personalized recommendations
python -m src.main --mode recommend

# Cohort analysis
python -m src.main --mode cohort

# Model monitoring (drift detection)
python -m src.main --mode monitor
```

#### Launch the Dashboard

```bash
streamlit run src/dashboard/app.py --server.port 8501
```

Then open http://localhost:8501 in your browser.

### Option 3: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific module tests
pytest tests/test_uplift_model.py -v
pytest tests/test_clv_model.py -v
pytest tests/test_ab_testing.py -v
```

---

## Configuration

All configurable parameters are centralized in `config/simulator_config.yaml`. Key sections include:

| Section | Key Parameters |
|---------|---------------|
| `simulation` | `num_customers: 20000`, `simulation_months: 12`, `random_seed: 42` |
| `churn_definition` | `no_purchase_days: 30`, `no_login_days: 60`, operator: OR |
| `treatment` | `treatment_ratio: 0.50`, `min_group_size: 10000` |
| `personas` | 6 customer types with distinct behavioral parameters |
| `ml_model` | `n_folds: 5`, `early_stopping_rounds: 20` |
| `dl_model` | Architecture (Transformer/LSTM), `hidden_size: 64`, `num_layers: 2` |
| `optimization` | `total_budget: 50000000` (KRW), 5 channel cost/limit configs |
| `monitoring` | PSI/KS thresholds for drift detection alerts |
| `small_mode` | Reduced settings for development (`5000` customers, `6` months) |

---

## Key Output Artifacts

After running the full pipeline, the following artifacts are generated:

| Artifact | Location | Description |
|----------|----------|-------------|
| CLV Predictions | `results/clv_predictions.csv` | Per-customer CLV with Top-20% flag |
| Segment Classification | `results/segments_6plus.csv` | 6+ segments with priority scores |
| Monitoring Report | `results/monitoring_report.json` | PSI/KS drift metrics & alerts |
| SHAP Summary | `results/shap_summary.png` | Top 10 feature importance plot |
| Budget Allocation | `results/budget_allocation.json` | Optimal channel-level budget split |
| Cohort Curves | `results/cohort_retention_curves.png` | M1/M3/M6/M12 retention curves |
| Trained Models | `models/` | Serialized ML, DL, uplift, CLV, survival models |

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Detailed system architecture, component diagrams, data flow |
| [Models](models.md) | Model training process, performance evaluation, comparison |
| [API Reference](api.md) | Module and class API documentation |
| [Deployment](deployment.md) | Docker deployment, production configuration |
| [Usage](usage.md) | Detailed usage guide and examples |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **ML Models** | XGBoost, LightGBM, scikit-learn, SHAP |
| **Deep Learning** | PyTorch 2.0+ (LSTM, Transformer) |
| **CLV / Survival** | lifelines (BG/NBD, Kaplan-Meier, Cox PH) |
| **Optimization** | scipy.optimize (linear programming) |
| **Statistics** | scipy.stats (Chi-square, Z-test, KS-test) |
| **Dashboard** | Streamlit, Plotly |
| **Experiment Tracking** | MLflow (SQLite backend) |
| **Streaming / Caching** | Redis 7 |
| **Containerization** | Docker Compose (4 services) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Configuration** | YAML (PyYAML) |
| **Testing** | pytest, pytest-cov (54+ test files) |

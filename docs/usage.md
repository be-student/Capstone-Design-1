# Usage Guide

> Complete guide for installing, configuring, and using the E-Commerce Customer Churn Prediction & Retention Optimization System.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [CLI Usage](#cli-usage)
4. [Docker Setup](#docker-setup)
5. [Streamlit Dashboard](#streamlit-dashboard)
6. [End-to-End Workflow](#end-to-end-workflow)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- **Python** 3.10 or higher
- **pip** (latest recommended)
- **Docker** 20.10+ and **Docker Compose** 2.0+ (for containerized deployment)
- **Git** (to clone the repository)

### Local Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd capstone

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -m pytest tests/ -v --tb=short
```

### Key Dependencies

| Category | Packages |
|----------|----------|
| Scientific Computing | numpy, pandas, scipy |
| Machine Learning | scikit-learn, xgboost, lightgbm |
| Deep Learning | torch (PyTorch 2.0+) |
| Explainability | shap |
| Survival Analysis / CLV | lifelines |
| Experiment Tracking | mlflow |
| Streaming | redis |
| Dashboard | streamlit, plotly |
| Visualization | matplotlib, seaborn |
| Configuration | pyyaml |
| Testing | pytest, pytest-cov |

---

## Configuration

All configurable parameters are defined in YAML files under `config/`.

### Main Configuration File

**`config/simulator_config.yaml`** — Central configuration for the entire system.

#### Simulation Settings

```yaml
simulation:
  n_customers: 20000       # Number of synthetic customers
  n_months: 12             # Observation window (months)
  random_seed: 42          # Reproducibility seed
```

#### Churn Definition

```yaml
churn_definition:
  no_purchase_days: 30     # Days without purchase to flag churn
  no_login_days: 60        # Days without login to flag churn
```

#### Model Settings

```yaml
ml_model:
  cv_folds: 5             # Cross-validation folds
  early_stopping_rounds: 20

dl_model:
  architecture: transformer  # Options: transformer, lstm
  sequence_window: 6
  hidden_dim: 64
  n_layers: 2
  n_heads: 4              # Transformer attention heads
```

#### Budget Optimization

```yaml
budget:
  total_budget_krw: 50000000  # 50M KRW default budget

optimization:
  channels:
    - email
    - sms
    - push_notification
    - coupon
    - call_center
```

#### Survival Analysis

```yaml
survival:
  penalizer: 0.01         # Cox PH regularization
  alpha: 0.05             # Significance level
```

#### Drift Detection

```yaml
drift_detection:
  psi_threshold_yellow: 0.10
  psi_threshold_red: 0.25

ks_drift_detection:
  warning_threshold: 0.05
  drift_threshold: 0.01
```

#### Customer Personas

The simulator generates customers from 6 behavioral personas:

| Persona | Share | Description |
|---------|-------|-------------|
| `vip_loyal` | 10% | VIP loyal customers |
| `regular_loyal` | 25% | Regular loyal customers |
| `bargain_hunter` | 20% | Price-sensitive shoppers |
| `new_customer` | 20% | Recently joined |
| `dormant` | 15% | Churning / inactive |
| `high_value_at_risk` | 10% | High-value but at risk |

#### MLflow Settings

```yaml
mlflow:
  tracking_uri: "http://localhost:5001"
  experiment_name: "churn_prediction"
```

#### Redis Settings

```yaml
redis:
  host: localhost
  port: 6379
```

### Overriding Configuration

You can copy and modify the configuration file to create custom profiles:

```bash
cp config/simulator_config.yaml config/my_config.yaml
# Edit config/my_config.yaml as needed
python src/main.py --mode all --config config/my_config.yaml
```

---

## CLI Usage

The CLI entrypoint is `src/main.py`, supporting **14 execution modes**.

### Basic Syntax

```bash
python src/main.py --mode <MODE> [--config <CONFIG_PATH>]
```

Or using module syntax:

```bash
python -m src.main --mode <MODE>
```

### Available Modes

#### Data Generation

```bash
# Generate synthetic customer data (20K customers, 12 months)
python src/main.py --mode simulate
```

Outputs customer profiles and event logs to `data/`.

#### Feature Engineering

```bash
# Compute 30+ features (RFM, behavioral, sequence)
python src/main.py --mode features
```

#### Model Training

```bash
# Train ML (XGBoost/LightGBM) + DL (Transformer/LSTM) + Ensemble
python src/main.py --mode train
```

Saves trained models to `models/` and logs metrics to MLflow.

#### Uplift Modeling

```bash
# Train T-Learner/S-Learner uplift models, 4-quadrant segmentation
python src/main.py --mode uplift
```

Identifies persuadables, sure things, lost causes, and sleeping dogs.

#### Customer Lifetime Value

```bash
# Predict CLV using BG/NBD + Gamma-Gamma models
python src/main.py --mode clv
```

Outputs CLV predictions with top-20% high-value customer flagging.

#### Budget Optimization

```bash
# LP-based budget allocation across marketing channels
python src/main.py --mode optimize
```

Allocates budget across email, SMS, push, coupon, and call center channels.

#### A/B Testing

```bash
# Statistical power analysis and significance testing
python src/main.py --mode ab_test
```

#### Survival Analysis

```bash
# Kaplan-Meier curves and Cox Proportional Hazards model
python src/main.py --mode survival
```

#### Personalized Recommendations

```bash
# Generate segment-specific retention actions
python src/main.py --mode recommend
```

#### Cohort Analysis

```bash
# Cohort retention curves and analysis
python src/main.py --mode cohort
```

#### Customer Segmentation

```bash
# RFM-based and K-means segmentation
python src/main.py --mode segment
```

#### Model Monitoring

```bash
# Run drift detection (PSI and KS-test)
python src/main.py --mode monitor
```

#### Streamlit Dashboard

```bash
# Launch the interactive dashboard on localhost:8501
python src/main.py --mode dashboard
```

#### Full Pipeline

```bash
# Execute all stages sequentially (end-to-end)
python src/main.py --mode all
```

Runs: simulate → features → train → uplift → clv → optimize → ab_test → survival → recommend → cohort → segment → monitor.

### Example Workflows

**Quick evaluation:**

```bash
# Generate data and train models
python src/main.py --mode simulate
python src/main.py --mode train

# Check model performance on the dashboard
python src/main.py --mode dashboard
```

**Full pipeline with custom config:**

```bash
python src/main.py --mode all --config config/simulator_config.yaml
```

### Output Artifacts

| Directory | Contents |
|-----------|----------|
| `data/` | Customer profiles, event logs (Parquet/CSV) |
| `models/` | Trained model files (joblib, PyTorch checkpoints) |
| `results/` | Analysis outputs (JSON, CSV, plots) |
| `mlflow/` | Experiment tracking data |

---

## Docker Setup

### Architecture Overview

The system runs as **4 Docker containers** orchestrated via Docker Compose:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  pipeline    │   │  dashboard  │   │   mlflow     │   │   redis     │
│  (ML/DL)    │──▶│  (Streamlit)│   │  (Tracking)  │   │  (Cache)    │
│              │   │  :8501      │   │  :5001       │   │  :6379      │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

### Quick Start with Docker

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up --build -d
```

### Service Details

| Service | Port | Description |
|---------|------|-------------|
| **pipeline** | — | Runs the ML/DL training pipeline |
| **dashboard** | 8501 | Streamlit interactive dashboard |
| **mlflow** | 5001 | MLflow tracking UI and model registry |
| **redis** | 6379 | In-memory data store for streaming |

### Common Docker Commands

```bash
# Build images without starting
docker-compose build

# Start specific services
docker-compose up dashboard mlflow

# View logs
docker-compose logs -f pipeline
docker-compose logs -f dashboard

# Stop all services
docker-compose down

# Stop and remove volumes (full cleanup)
docker-compose down -v

# Rebuild a single service
docker-compose build pipeline
docker-compose up -d pipeline

# Check service health
docker-compose ps
```

### Running Specific Pipeline Modes in Docker

```bash
# Override the default command to run a specific mode
docker-compose run pipeline python -m src.main --mode simulate
docker-compose run pipeline python -m src.main --mode train
docker-compose run pipeline python -m src.main --mode uplift
```

### Persistent Volumes

| Volume | Mount Point | Purpose |
|--------|-------------|---------|
| `mlflow-data` | `/mlflow` | Experiment tracking database and artifacts |
| `redis-data` | `/data` | Redis persistence (AOF) |
| `pipeline-output` | `/app/output` | Pipeline results and model artifacts |

### Environment Variables

The Docker Compose configuration sets these automatically:

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL (internal) |
| `REDIS_HOST` | `redis` | Redis hostname (internal) |
| `PYTHONPATH` | `/app` | Python module resolution path |

### Health Checks

All services include health checks:

- **MLflow**: HTTP GET `http://localhost:5000/health`
- **Redis**: `redis-cli ping`
- **Dashboard**: HTTP GET `http://localhost:8501/_stcore/health`

### Network

All containers communicate on the `churn-network` bridge network. Internal service names (`mlflow`, `redis`, `pipeline`, `dashboard`) resolve automatically within this network.

---

## Streamlit Dashboard

### Launching the Dashboard

**Locally:**

```bash
# Via CLI entrypoint
python src/main.py --mode dashboard

# Or directly with Streamlit
streamlit run src/dashboard/app.py --server.port 8501
```

**Via Docker:**

```bash
docker-compose up dashboard
# Access at http://localhost:8501
```

### Dashboard Pages

The dashboard provides **10 interactive views** accessible from the sidebar navigation:

#### 1. Overview

- KPI summary cards (total customers, churn rate, average CLV)
- Churn distribution chart
- Risk level breakdown
- Segment-wise churn rates
- Feature importance rankings
- Individual customer lookup

#### 2. Cohort Analysis

- Monthly cohort retention curves
- Retention heatmaps
- Cohort comparison metrics

#### 3. Uplift Analysis

- 4-quadrant segmentation visualization (persuadables, sure things, lost causes, sleeping dogs)
- Uplift distribution plots
- Treatment effect analysis

#### 4. Customer Lifetime Value

- CLV distribution across segments
- Top-20% high-value customer identification
- CLV vs. churn risk scatter plot

#### 5. Budget Optimization

- Optimal budget allocation across channels
- What-if scenario analysis
- ROI projections per channel

#### 6. A/B Testing

- Experiment results and statistical significance
- Power analysis curves
- Effect size visualization

#### 7. Survival Analysis

- Kaplan-Meier survival curves by segment
- Median survival time comparison
- Cox PH hazard ratios

#### 8. Recommendations

- Segment-specific retention actions
- Personalized intervention strategies
- Budget-aware recommendation rankings

#### 9. Monitoring

- Feature drift detection (PSI scores)
- KS-test drift alerts
- Drift trend over time

#### 10. System Health

- Pipeline execution state and checkpoints
- Service connectivity status
- Recent pipeline run history

### Dashboard Data Requirements

The dashboard reads output files from `data/`, `results/`, and `models/`. Run the pipeline first to generate the required data:

```bash
# Generate all data the dashboard needs
python src/main.py --mode all

# Then launch the dashboard
python src/main.py --mode dashboard
```

---

## End-to-End Workflow

### Recommended Steps

```bash
# Step 1: Set up configuration
# Review and customize config/simulator_config.yaml

# Step 2: Generate synthetic data
python src/main.py --mode simulate

# Step 3: Engineer features
python src/main.py --mode features

# Step 4: Train churn prediction models
python src/main.py --mode train

# Step 5: Run analytical modules
python src/main.py --mode uplift
python src/main.py --mode clv
python src/main.py --mode survival
python src/main.py --mode cohort
python src/main.py --mode segment

# Step 6: Optimize and test
python src/main.py --mode optimize
python src/main.py --mode ab_test
python src/main.py --mode recommend

# Step 7: Monitor model health
python src/main.py --mode monitor

# Step 8: Visualize results
python src/main.py --mode dashboard
```

Or run everything at once:

```bash
python src/main.py --mode all
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test modules
python -m pytest tests/test_uplift.py -v
python -m pytest tests/test_clv.py -v
python -m pytest tests/test_main_cli.py -v
```

---

## Troubleshooting

### Common Issues

**Port conflicts:**

```bash
# Check if ports are in use
lsof -i :8501  # Dashboard
lsof -i :5001  # MLflow
lsof -i :6379  # Redis
```

**Docker build failures:**

```bash
# Clear Docker cache and rebuild
docker-compose build --no-cache
```

**Missing data files:**

If the dashboard shows empty views, ensure the pipeline has been run:

```bash
python src/main.py --mode simulate
python src/main.py --mode all
```

**MLflow connection errors:**

When running locally, make sure the MLflow tracking URI matches your setup:

```yaml
# config/simulator_config.yaml
mlflow:
  tracking_uri: "http://localhost:5001"  # Local
  # tracking_uri: "http://mlflow:5000"   # Docker
```

**Redis connection errors:**

Redis is optional for local development. The streaming module gracefully degrades if Redis is unavailable. For Docker, Redis starts automatically via `docker-compose up`.

**Memory issues with large datasets:**

Reduce the customer count in configuration:

```yaml
simulation:
  n_customers: 5000  # Reduce from 20000 for testing
```

### Getting Help

- **Architecture details:** See [docs/architecture.md](architecture.md)
- **Model documentation:** See [docs/models.md](models.md)
- **API reference:** See [docs/api.md](api.md)
- **Deployment guide:** See [docs/deployment.md](deployment.md)

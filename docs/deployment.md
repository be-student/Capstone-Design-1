# Deployment Guide

> **Customer Churn Prediction & Retention Optimization System — Deployment, CLI, and Dashboard Reference**

This document covers Docker setup instructions, CLI usage guide, Streamlit dashboard
configuration, and production deployment guidelines for the e-commerce churn
prediction and retention optimization system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Docker Setup Instructions](#2-docker-setup-instructions)
   - 2.1 [Service Architecture](#21-service-architecture)
   - 2.2 [Quick Start](#22-quick-start)
   - 2.3 [Docker Compose Configuration](#23-docker-compose-configuration)
   - 2.4 [Dockerfiles](#24-dockerfiles)
   - 2.5 [Environment Variables](#25-environment-variables)
   - 2.6 [Volumes & Networking](#26-volumes--networking)
   - 2.7 [Service Dependencies & Health Checks](#27-service-dependencies--health-checks)
3. [CLI Usage Guide](#3-cli-usage-guide)
   - 3.1 [Overview](#31-overview)
   - 3.2 [Command Syntax](#32-command-syntax)
   - 3.3 [Available Modes](#33-available-modes)
   - 3.4 [Flags & Options](#34-flags--options)
   - 3.5 [Usage Examples](#35-usage-examples)
   - 3.6 [Running via Docker](#36-running-via-docker)
   - 3.7 [Running via Python Module](#37-running-via-python-module)
   - 3.8 [Full Pipeline (`--mode all`)](#38-full-pipeline---mode-all)
4. [Streamlit Dashboard Configuration](#4-streamlit-dashboard-configuration)
   - 4.1 [Dashboard Views](#41-dashboard-views)
   - 4.2 [Launching the Dashboard](#42-launching-the-dashboard)
   - 4.3 [Configuration Options](#43-configuration-options)
   - 4.4 [Data Sources](#44-data-sources)
   - 4.5 [Dashboard Customization](#45-dashboard-customization)
5. [Production Deployment Guidelines](#5-production-deployment-guidelines)
   - 5.1 [Production Migration Path](#51-production-migration-path)
   - 5.2 [Scaling Considerations](#52-scaling-considerations)
   - 5.3 [Resource Limits](#53-resource-limits)
   - 5.4 [Security Hardening](#54-security-hardening)
   - 5.5 [Monitoring & Observability](#55-monitoring--observability)
   - 5.6 [Backup & Recovery](#56-backup--recovery)
6. [Troubleshooting](#6-troubleshooting)
7. [Quick Reference](#7-quick-reference)

---

## 1. Prerequisites

| Requirement | Minimum Version | Notes |
|------------|----------------|-------|
| **Docker** | 20.10+ | `docker --version` to verify |
| **Docker Compose** | 2.0+ (V2) | `docker compose version` to verify |
| **Python** | 3.10+ | For local (non-Docker) development |
| **Disk Space** | 10 GB free | For images, data, models, and MLflow artifacts |
| **RAM** | 8 GB recommended | Pipeline container is memory-intensive during training |
| **CPU** | 4 cores recommended | All models run on CPU only (no GPU required) |
| **OS** | Linux, macOS, Windows (WSL2) | Docker Desktop on macOS/Windows |

Verify your installation:

```bash
docker --version          # Docker >= 20.10
docker compose version    # Compose >= 2.0
docker info               # Confirm daemon is running
python --version          # Python >= 3.10 (for local dev)
```

---

## 2. Docker Setup Instructions

### 2.1 Service Architecture

The system comprises **4 containers** orchestrated by Docker Compose:

| Service | Image Base | Default Port | Purpose |
|---------|-----------|-------------|---------|
| `mlflow` | `python:3.10-slim` | 5001 (host) → 5000 (container) | MLflow experiment tracking (SQLite backend) |
| `redis` | `redis:7-alpine` | 6379 | Real-time feature store, caching |
| `pipeline` | `python:3.10-slim` | — | ML/DL training, data simulation, optimization |
| `dashboard` | `python:3.10-slim` | 8501 | Streamlit interactive dashboard |

```
                Host Machine
                ============
                     |
      +--------------+--------------+
      |              |              |
 localhost:8501 localhost:5001 localhost:6379
      |              |              |
┌─────┴──────┐ ┌────┴─────┐ ┌──────┴──────┐
│  dashboard │ │  mlflow  │ │    redis    │
│ (Streamlit)│ │(Tracking)│ │  (Cache)    │
└─────┬──────┘ └────┬─────┘ └──────┬──────┘
      |              |              |
      +--------------+--------------+
                     |
           Docker Bridge Network
            (churn-network)
                     |
              ┌──────┴──────┐
              │   pipeline  │
              │  (ML/DL)    │
              └─────────────┘
```

### 2.2 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/capstone.git
cd capstone

# 2. (Optional) Create .env to override defaults
cat > .env <<'EOF'
PIPELINE_MODE=all
SMALL=false
VERBOSE=false
EOF

# 3. Build and start all services
docker compose up --build

# 4. Monitor pipeline progress
docker compose logs -f pipeline

# 5. Access services once pipeline completes
#    Dashboard:  http://localhost:8501
#    MLflow UI:  http://localhost:5001
```

**Common quick-start scenarios:**

```bash
# Full pipeline with small dataset (fastest)
SMALL=true docker compose up --build

# Dashboard only (skip pipeline, assumes data exists)
SKIP_PIPELINE=true docker compose up dashboard

# Train only
PIPELINE_MODE=train docker compose up pipeline

# Detached mode (background)
docker compose up -d --build

# Stop everything
docker compose down

# Full cleanup (remove volumes & images)
docker compose down -v --rmi local
```

### 2.3 Docker Compose Configuration

The `docker-compose.yml` defines four services with the following key configuration:

**MLflow Tracking Server:**
- SQLite backend store (no PostgreSQL required)
- Filesystem artifact store at `/mlflow/artifacts`
- Health check: `curl -f http://localhost:5001/health` from the host, or `curl -f http://mlflow:5000/health` inside Docker
- Restart policy: `unless-stopped`

**Redis:**
- `redis:7-alpine` with AOF persistence enabled
- Memory limit: 256 MB with LRU eviction
- Health check: `redis-cli ping`

**Pipeline:**
- Runs the CLI entrypoint via `scripts/pipeline_entrypoint.sh`
- Waits for both MLflow and Redis to be healthy before starting
- Volumes bind-mount `config/`, `data/`, `src/`, `models/`, `results/`

**Dashboard:**
- Streamlit on port 8501 with headless mode enabled
- Waits for pipeline to complete (`service_completed_successfully`)
- Set `SKIP_PIPELINE=true` to skip waiting for pipeline
- Health check: `curl -f http://localhost:8501/_stcore/health`

### 2.4 Dockerfiles

The project includes three Dockerfiles:

| Dockerfile | Purpose | Key Features |
|-----------|---------|-------------|
| `Dockerfile.pipeline` | ML/DL training pipeline | `build-essential` for native extensions, `libgomp1` for OpenMP, configurable via env vars |
| `Dockerfile.dashboard` | Streamlit dashboard | Minimal dependencies (no build tools), exposes port 8501, headless Streamlit |
| `Dockerfile.mlflow` | MLflow tracking server | MLflow 2.12.1, SQLite backend, lightweight |

**Building individual images:**

```bash
docker build -f Dockerfile.pipeline -t churn-pipeline .
docker build -f Dockerfile.dashboard -t churn-dashboard .
docker build -f Dockerfile.mlflow -t churn-mlflow .
```

### 2.5 Environment Variables

#### Pipeline Container

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_MODE` | `all` | Execution mode (see [CLI modes](#33-available-modes)) |
| `SMALL` | `false` | `"true"` for reduced dataset (5K customers, 6 months) |
| `BUDGET` | _(empty)_ | Budget cap for `optimize` mode (KRW) |
| `VERBOSE` | `false` | `"true"` for DEBUG-level logging |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `PYTHONPATH` | `/app` | Python module search path |

#### Dashboard Container

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Streamlit bind address |
| `SKIP_PIPELINE` | `false` | `"true"` to skip waiting for pipeline |
| `PIPELINE_MAX_RETRIES` | `1` | Max retries waiting for pipeline |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `REDIS_HOST` | `redis` | Redis hostname |

#### Port Overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_PORT` | `5001` | Host port for MLflow UI |
| `REDIS_PORT` | `6379` | Host port for Redis |
| `DASHBOARD_PORT` | `8501` | Host port for Streamlit dashboard |

#### Override via .env File

Create a `.env` file in the project root:

```bash
# .env
PIPELINE_MODE=all
SMALL=true
BUDGET=50000000
VERBOSE=false
DASHBOARD_PORT=8501
MLFLOW_PORT=5001
```

Docker Compose automatically loads this file. Use a custom env file:

```bash
docker compose --env-file .env.production up --build
```

### 2.6 Volumes & Networking

**Named Docker Volumes:**

| Volume | Mount Point | Purpose |
|--------|-----------|---------|
| `mlflow-data` | `/mlflow` | MLflow database and artifacts |
| `redis-data` | `/data` | Redis AOF persistence |
| `pipeline-output` | `/app/output` | Pipeline output artifacts |

**Bind Mounts (shared between pipeline & dashboard):**

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./config/` | `/app/config/` | YAML configuration files |
| `./data/` | `/app/data/` | Raw & processed data |
| `./src/` | `/app/src/` | Source code (live reload in dev) |
| `./models/` | `/app/models/` | Trained model artifacts |
| `./results/` | `/app/results/` | Output CSVs, plots, reports |

**Networking:**

All services communicate over the `churn-network` Docker bridge network.
Services reference each other by service name (e.g., `redis:6379`, `mlflow:5000`).

### 2.7 Service Dependencies & Health Checks

**Startup Order:**

```
1. redis     ─── starts first (no dependencies)
2. mlflow    ─── starts in parallel with redis (no dependencies)
3. pipeline  ─── waits for redis [healthy] AND mlflow [healthy]
4. dashboard ─── waits for redis [healthy], mlflow [healthy], AND pipeline [completed]
```

**Health Checks:**

| Service | Check | Interval | Timeout | Retries | Start Period |
|---------|-------|----------|---------|---------|-------------|
| `redis` | `redis-cli ping` | 10s | 5s | 3 | — |
| `mlflow` | `curl -f http://localhost:5001/health` (host) / `curl -f http://mlflow:5000/health` (Docker network) | 15s | 10s | 5 | 10s |
| `dashboard` | `curl -f http://localhost:8501/_stcore/health` | 30s | 10s | 3 | 20s |

**Failure Scenarios:**

| Scenario | Behavior |
|----------|----------|
| Redis fails | Pipeline and dashboard will not start |
| MLflow fails | Pipeline waits for health check; retries up to 5 times |
| Pipeline crashes | Dashboard will not launch (depends on `service_completed_successfully`) |
| Pipeline crashes + `SKIP_PIPELINE=true` | Dashboard launches regardless |
| Dashboard crashes | Auto-restarts via `restart: unless-stopped` |

---

## 3. CLI Usage Guide

### 3.1 Overview

The CLI entrypoint (`src/main.py`) is the primary interface for running all system
components. It supports 14 execution modes covering data generation, model training,
analysis, optimization, and visualization.

### 3.2 Command Syntax

```bash
python src/main.py --mode <MODE> [OPTIONS]

# Or as a Python module:
python -m src.main --mode <MODE> [OPTIONS]
```

### 3.3 Available Modes

| Mode | Description | Prerequisites |
|------|-------------|--------------|
| `simulate` | Generate synthetic customer data | None |
| `train` | Train ML, DL, and Ensemble churn models | `simulate` (data in `data/raw/`) |
| `uplift` | Train uplift model (T-Learner/S-Learner) and 4-quadrant segmentation | `simulate` |
| `clv` | Predict Customer Lifetime Value with ML regression and holdout validation | `simulate` |
| `optimize` | LP-based budget optimization across segments | `uplift` + `clv` (or uses synthetic data) |
| `ab_test` | A/B test power analysis and significance testing | `simulate` (uses treatment groups) |
| `survival` | Cox Proportional Hazards survival analysis | `simulate` + features |
| `recommend` | Generate personalized retention recommendations | `simulate` (+ optional `uplift`, `clv`) |
| `cohort` | Cohort retention analysis with heatmaps | `simulate` (events data) |
| `segment` | Customer segmentation using churn probability, uplift, and CLV | `simulate` + `train` + `uplift` + `clv` |
| `features` | Run feature engineering pipeline only | `simulate` |
| `monitor` | Model monitoring (PSI & KS drift detection) | `simulate` + features |
| `dashboard` | Launch Streamlit dashboard (localhost:8501) | Results from other modes |
| `all` | Run full end-to-end pipeline with checkpoint/resume | None |

### 3.4 Flags & Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | str | _(required)_ | Execution mode (see table above) |
| `--config` | str | `config/simulator_config.yaml` | Path to YAML configuration file |
| `--data` | str | `data/raw/` | Data directory for input files |
| `--output` | str | _(auto)_ | Output base directory (creates `results/` and `models/` subdirs) |
| `--budget` | int | _(from config)_ | Total marketing budget in KRW (for `optimize` mode) |
| `--small` | flag | `false` | Enable small mode: 5,000 customers, 6 months |
| `--learner` | str | `t_learner` | Uplift learner type: `t_learner` or `s_learner` |
| `--cohort-type` | str | `monthly` | Cohort type: `monthly`, `weekly`, or `behavioral` |
| `-v, --verbose` | flag | `false` | Enable DEBUG logging level |
| `-q, --quiet` | flag | `false` | Suppress output (WARNING level only) |

### 3.5 Usage Examples

**Data Simulation:**

```bash
# Generate full dataset (20,000 customers, 12 months)
python src/main.py --mode simulate

# Generate small dataset (5,000 customers, 6 months)
python src/main.py --mode simulate --small

# Custom config
python src/main.py --mode simulate --config config/custom_config.yaml
```

**Model Training:**

```bash
# Train ML, DL, and Ensemble models
python src/main.py --mode train

# Train with verbose logging
python src/main.py --mode train -v

# Train on small dataset
python src/main.py --mode train --small
```

**Uplift Modeling:**

```bash
# T-Learner uplift model (default)
python src/main.py --mode uplift

# S-Learner uplift model
python src/main.py --mode uplift --learner s_learner
```

**Budget Optimization:**

```bash
# Optimize with default budget from config
python src/main.py --mode optimize

# Optimize with custom budget (50M KRW)
python src/main.py --mode optimize --budget 50000000

# Optimize with 100M KRW budget
python src/main.py --mode optimize --budget 100000000
```

**Analysis Modes:**

```bash
# CLV prediction
python src/main.py --mode clv

# A/B test analysis
python src/main.py --mode ab_test

# Survival analysis (Cox PH)
python src/main.py --mode survival

# Cohort retention analysis (monthly)
python src/main.py --mode cohort --cohort-type monthly

# Weekly cohort analysis
python src/main.py --mode cohort --cohort-type weekly
```

**Monitoring & Other:**

```bash
# Model drift detection
python src/main.py --mode monitor

# Feature engineering only
python src/main.py --mode features

# Customer segmentation
python src/main.py --mode segment

# Personalized recommendations
python src/main.py --mode recommend
```

**Dashboard:**

```bash
# Launch Streamlit dashboard
python src/main.py --mode dashboard
```

### 3.6 Running via Docker

The pipeline container uses `scripts/pipeline_entrypoint.sh` to translate environment
variables into CLI flags:

```bash
# Via environment variables (Docker Compose style)
PIPELINE_MODE=train docker compose up pipeline
PIPELINE_MODE=optimize BUDGET=50000000 docker compose up pipeline
PIPELINE_MODE=simulate SMALL=true docker compose up pipeline

# Via direct CLI override (docker compose run)
docker compose run pipeline python -m src.main --mode train
docker compose run pipeline python -m src.main --mode optimize --budget 50000000
docker compose run pipeline python -m src.main --mode simulate --small -v

# Full pipeline with small dataset
SMALL=true docker compose up --build
```

**Environment-to-CLI mapping in `pipeline_entrypoint.sh`:**

| Environment Variable | CLI Flag | Example |
|---------------------|---------|---------|
| `PIPELINE_MODE=train` | `--mode train` | `PIPELINE_MODE=train docker compose up pipeline` |
| `SMALL=true` | `--small` | `SMALL=true docker compose up pipeline` |
| `BUDGET=50000000` | `--budget 50000000` | `BUDGET=50000000 docker compose up pipeline` |
| `VERBOSE=true` | `-v` | `VERBOSE=true docker compose up pipeline` |

### 3.7 Running via Python Module

```bash
# Direct invocation
python src/main.py --mode train

# As a module (works from project root)
python -m src.main --mode train

# The __main__.py entrypoint enables:
python -m src --mode train
```

### 3.8 Full Pipeline (`--mode all`)

The `all` mode runs the complete end-to-end pipeline in this order:

```
1. data_generation       → simulate customer data
2. preprocessing         → initial data prep
3. feature_engineering   → compute feature matrix
4. ml_model_training     → train ML models (XGBoost, LightGBM)
5. dl_model_training     → train deep learning model (PyTorch)
6. ensemble_creation     → build ensemble from ML + DL
7. uplift_modeling       → T-Learner uplift model
8. clv_prediction        → ML-based lifetime value
9. customer_segmentation → churn/uplift/CLV 6+ segments
10. budget_optimization  → LP-based budget allocation
11. recommendations      → personalized retention actions
12. cohort_analysis      → cohort retention and journey analysis
13. ab_testing           → A/B test statistical analysis
14. survival_analysis    → Cox PH survival curves
15. scoring_api_setup    → model monitoring
16. mlflow_logging       → experiment tracking
```

**Checkpoint/Resume:** The pipeline stores state in `data/raw/pipeline_state.json`.
On restart, completed stages are automatically skipped:

```bash
# First run (may fail at step 8)
python src/main.py --mode all --small

# Resume from last checkpoint (steps 1-7 are skipped)
python src/main.py --mode all --small

# Force re-run from scratch
rm data/raw/pipeline_state.json
python src/main.py --mode all --small
```

**Output:** Each mode handler returns a JSON result to stdout:

```json
{
  "mode": "train",
  "status": "completed",
  "ml_metrics": {"auc_roc": 0.862, "accuracy": 0.84},
  "dl_metrics": {"auc_roc": 0.847, "accuracy": 0.82}
}
```

---

## 4. Streamlit Dashboard Configuration

### 4.1 Dashboard Views

The Streamlit dashboard (`src/dashboard/app.py`) provides interactive views for all
system outputs. Views are accessible via a sidebar navigation.

| View | Description | Data Source |
|------|-------------|------------|
| **Churn Overview** | KPI cards, churn rate, risk distribution, top features | `results/model_metrics.json`, features |
| **Model Performance** | ML vs DL vs Ensemble comparison, ROC curves, confusion matrices | `results/model_metrics.json`, `models/` |
| **Uplift Modeling** | 4-quadrant segmentation, treatment effect distribution | `results/uplift_results.csv` |
| **CLV Analysis** | CLV distribution, top customers, segment breakdown | `results/clv_predictions.csv` |
| **A/B Testing** | Power analysis, significance results, confidence intervals | `results/ab_test_results.json` |
| **Survival Analysis** | Kaplan-Meier curves, Cox PH hazard ratios | `results/survival_results.json` |
| **Cohort Retention** | Retention heatmap, cohort curves | `results/cohort_retention_matrix.csv` |
| **Budget Optimization** | Allocation by segment, what-if scenarios | `results/budget_optimization.csv` |
| **Recommendations** | Per-customer retention actions, priority ranking | `results/recommendations.csv` |
| **Model Monitoring** | PSI/KS drift scores, performance trends | `results/monitoring_report.json` |
| **System Health** | Container status, pipeline state, resource usage | Pipeline state, Redis |

### 4.2 Launching the Dashboard

**Via CLI:**

```bash
# Launch from project root (localhost:8501)
python src/main.py --mode dashboard

# Or directly with Streamlit
streamlit run src/dashboard/app.py \
  --server.port 8501 \
  --server.address localhost \
  --server.headless true
```

**Via Docker Compose:**

```bash
# Dashboard with full pipeline
docker compose up --build

# Dashboard only (skip pipeline)
SKIP_PIPELINE=true docker compose up dashboard

# Custom port
DASHBOARD_PORT=9000 docker compose up dashboard
```

**Via Docker standalone:**

```bash
docker build -f Dockerfile.dashboard -t churn-dashboard .
docker run -p 8501:8501 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  churn-dashboard
```

### 4.3 Configuration Options

The dashboard reads all configuration from `config/simulator_config.yaml`:

```yaml
# Key dashboard-relevant configuration sections:

simulation:
  random_seed: 42
  num_customers: 20000

churn_definition:
  no_purchase_days: 30
  no_login_days: 60
  operator: "OR"

treatment:
  treatment_ratio: 0.50
```

**Streamlit server configuration** can be set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Server port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Bind address (`localhost` for local dev) |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Don't auto-open browser |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disable telemetry |

Or via `.streamlit/config.toml` in the project root:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 4.4 Data Sources

The dashboard loads data through `src/dashboard/data_loader.py`, which reads from:

| Directory | File(s) | View(s) |
|-----------|---------|---------|
| `results/` | `model_metrics.json` | Churn Overview, Model Performance |
| `results/` | `uplift_results.csv` | Uplift Modeling |
| `results/` | `clv_predictions.csv` | CLV Analysis |
| `results/` | `ab_test_results.json` | A/B Testing |
| `results/` | `survival_results.json` | Survival Analysis |
| `results/` | `cohort_retention_matrix.csv` | Cohort Retention |
| `results/` | `budget_optimization.csv` | Budget Optimization |
| `results/` | `recommendations.csv` | Recommendations |
| `results/` | `monitoring_report.json` | Model Monitoring |
| `models/` | `*.pkl`, `*.pt` | Model artifacts |
| `data/raw/` | `customers.parquet`, `events.parquet` | Raw data views |
| `config/` | `simulator_config.yaml` | System configuration |

> **Note:** If a result file is missing, the dashboard degrades gracefully by
> showing a "No data available" message for that view. Run the corresponding
> CLI mode to generate the data.

### 4.5 Dashboard Customization

**Adding new views:**

1. Create a new view module in `src/dashboard/` (e.g., `custom_view.py`)
2. Define a `render_custom_view(config, data_loader)` function
3. Import and register in `src/dashboard/app.py`
4. Add to the `PAGES` list in `src/dashboard/utils/dashboard_helpers.py`

**Helper utilities** available in `src/dashboard/utils/dashboard_helpers.py`:

| Function | Purpose |
|----------|---------|
| `format_currency(value)` | Format KRW currency values |
| `format_percentage(value)` | Format percentages with 1 decimal |
| `format_count(value)` | Format large numbers with commas |
| `classify_risk(prob)` | Classify churn probability into risk levels |
| `get_risk_color(risk_level)` | Get color for risk visualization |
| `get_color_palette()` | Get consistent color palette |
| `get_segment_colors()` | Get segment-specific colors |
| `build_sidebar_info(config)` | Build sidebar metadata |
| `validate_predictions(df)` | Validate prediction dataframes |

---

## 5. Production Deployment Guidelines

### 5.1 Production Migration Path

| Component | Development (Current) | Production (Recommended) |
|-----------|----------------------|--------------------------|
| **MLflow Backend** | SQLite (`sqlite:////mlflow/mlflow.db`) | PostgreSQL (`postgresql://...`) |
| **MLflow Artifacts** | Local filesystem (`/mlflow/artifacts`) | S3 / MinIO (`s3://bucket/artifacts`) |
| **Redis** | Single node, no auth, 256 MB | Redis Sentinel/Cluster with AUTH, 2+ GB |
| **API Auth** | None | OAuth2 / JWT tokens |
| **Secrets** | `.env` file | Docker Secrets / HashiCorp Vault |
| **Logging** | stdout/stderr | ELK Stack / CloudWatch / Loki |
| **Monitoring** | File-based (`monitoring_report.json`) | Prometheus + Grafana |
| **Orchestration** | Docker Compose | Kubernetes (Helm charts) |
| **CI/CD** | Manual `docker compose up` | GitHub Actions + ArgoCD |
| **TLS** | None | Terminate at reverse proxy (nginx/traefik) |

### 5.2 Scaling Considerations

**Horizontal Scaling:**

| Component | Strategy |
|-----------|---------|
| **Dashboard** | Stateless — run N replicas behind a load balancer |
| **Pipeline** | Batch job — scale vertically (more CPU/RAM); for distributed training use Ray/Dask |
| **Redis** | Redis Cluster or Sentinel for HA; current single-node sufficient for < 100K customers |
| **MLflow** | Migrate to PostgreSQL backend; S3 for artifacts |

**Example: Dashboard behind nginx:**

```yaml
# docker-compose.override.yml
services:
  dashboard:
    deploy:
      replicas: 3
    ports: []  # Remove host port; use load balancer

  nginx:
    image: nginx:alpine
    ports:
      - "8501:80"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - dashboard
```

### 5.3 Resource Limits

**Recommended resource allocation by dataset size:**

| Dataset Size | Pipeline RAM | Pipeline CPU | Total System RAM |
|-------------|-------------|-------------|------------------|
| 5K customers (small) | 2 GB | 2 cores | 6 GB |
| 20K customers (default) | 4–6 GB | 4 cores | 10 GB |
| 100K customers | 16 GB | 8 cores | 24 GB |
| 500K+ customers | 32 GB+ | 16 cores | 48 GB+ |

**Docker Compose resource limits:**

```yaml
# docker-compose.override.yml
services:
  pipeline:
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 8G
        reservations:
          cpus: "2.0"
          memory: 4G

  dashboard:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G

  redis:
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 1G

  mlflow:
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 1G
```

### 5.4 Security Hardening

**Network Isolation:**

```yaml
# docker-compose.prod.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  redis:
    networks: [backend]
  mlflow:
    networks: [backend]
  pipeline:
    networks: [frontend, backend]
  dashboard:
    networks: [frontend, backend]
```

**Redis Authentication:**

```yaml
redis:
  command: redis-server --appendonly yes --requirepass "${REDIS_PASSWORD}"
  environment:
    - REDIS_PASSWORD=${REDIS_PASSWORD}
```

**Non-root containers:**

```dockerfile
# Add to Dockerfiles
RUN useradd -m -r appuser
USER appuser
```

**Sensitive files — never commit to version control:**

```gitignore
# .gitignore
.env
.env.production
config/secrets.yaml
*.pem
*.key
```

### 5.5 Monitoring & Observability

**Container health monitoring:**

```bash
# Container status (includes health checks)
docker compose ps

# Real-time resource usage
docker stats churn-pipeline churn-dashboard churn-redis churn-mlflow

# Service logs
docker compose logs -f --tail=100
docker compose logs -f pipeline
docker compose logs -f dashboard
```

**Pipeline state monitoring:**

The pipeline tracks execution state in `data/raw/pipeline_state.json`:

```bash
# Check pipeline progress
cat data/raw/pipeline_state.json | python -m json.tool

# From within Docker
docker compose exec pipeline cat /app/data/raw/pipeline_state.json
```

**MLflow experiment tracking:**

```bash
# Access MLflow UI
open http://localhost:5001

# Query metrics via API
curl -s http://localhost:5001/api/2.0/mlflow/experiments/search | python -m json.tool
```

**Model drift monitoring:**

The `monitor` mode computes PSI and KS statistics for detecting data drift.
Results are saved to `results/monitoring_report.json` and displayed on the
dashboard's Model Monitoring view.

**Structured logging (production):**

```yaml
# docker-compose.override.yml
services:
  pipeline:
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
        tag: "{{.Name}}"

  dashboard:
    logging:
      driver: json-file
      options:
        max-size: "20m"
        max-file: "3"
```

### 5.6 Backup & Recovery

**Data backup:**

```bash
# Backup all persistent data
tar czf backup_$(date +%Y%m%d).tar.gz data/ models/ results/ config/

# Backup MLflow data
docker compose exec mlflow tar czf /tmp/mlflow_backup.tar.gz /mlflow
docker compose cp mlflow:/tmp/mlflow_backup.tar.gz ./backups/

# Backup Redis
docker compose exec redis redis-cli BGSAVE
docker compose cp redis:/data/appendonly.aof ./backups/
```

**Recovery:**

```bash
# Restore data
tar xzf backup_20260321.tar.gz

# Restart services
docker compose down
docker compose up --build
```

---

## 6. Troubleshooting

### Common Issues

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Redis container exits | Port 6379 already in use | `lsof -i :6379` and stop conflicting process, or set `REDIS_PORT=6380` in `.env` |
| Pipeline OOM killed | Insufficient memory | Increase Docker memory limit or use `SMALL=true` |
| MLflow UI shows no experiments | Volume permissions | Check `mlflow-data` volume; restart `mlflow` service |
| Dashboard "No data available" | Pipeline hasn't completed | Check pipeline logs: `docker compose logs pipeline` |
| Dashboard won't start | Waiting for pipeline | Set `SKIP_PIPELINE=true` in `.env` |
| Port conflicts | Services on default ports | Override ports via `DASHBOARD_PORT`, `MLFLOW_PORT`, `REDIS_PORT` in `.env` |
| Import errors | `PYTHONPATH` not set | Ensure `PYTHONPATH=/app` in container or run from project root |
| `FileNotFoundError: No customer data` | Data not generated | Run `--mode simulate` first |

### Diagnostic Commands

```bash
# Full system status
docker compose ps -a

# Check resource usage
docker stats --no-stream

# Shell into containers
docker compose exec pipeline bash
docker compose exec redis redis-cli

# Test Redis connectivity
docker compose exec pipeline python -c "import redis; r=redis.Redis('redis'); print(r.ping())"

# Test MLflow connectivity
docker compose exec pipeline python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); print(mlflow.get_tracking_uri())"

# Check disk usage
docker system df -v
```

### Reset and Clean Restart

```bash
# Stop everything
docker compose down

# Remove generated data
rm -rf data/raw/* models/* results/*

# Remove Docker volumes
docker compose down -v

# Remove Docker images (force rebuild)
docker compose down --rmi local

# Fresh start
docker compose up --build
```

---

## 7. Quick Reference

```bash
# ──── Lifecycle ────
docker compose up --build              # Build and start all services
docker compose up -d --build           # Detached mode
docker compose down                    # Stop all services
docker compose down -v                 # Stop and remove volumes
docker compose restart dashboard       # Restart specific service

# ──── Pipeline Modes (via env vars) ────
PIPELINE_MODE=simulate SMALL=true docker compose up pipeline
PIPELINE_MODE=train docker compose up pipeline
PIPELINE_MODE=uplift docker compose up pipeline
PIPELINE_MODE=clv docker compose up pipeline
PIPELINE_MODE=optimize BUDGET=50000000 docker compose up pipeline
PIPELINE_MODE=ab_test docker compose up pipeline
PIPELINE_MODE=survival docker compose up pipeline
PIPELINE_MODE=recommend docker compose up pipeline
PIPELINE_MODE=cohort docker compose up pipeline
PIPELINE_MODE=all SMALL=true docker compose up --build

# ──── Pipeline Modes (via CLI) ────
python src/main.py --mode simulate --small
python src/main.py --mode train
python src/main.py --mode uplift --learner t_learner
python src/main.py --mode clv
python src/main.py --mode optimize --budget 50000000
python src/main.py --mode ab_test
python src/main.py --mode survival
python src/main.py --mode recommend
python src/main.py --mode cohort --cohort-type monthly
python src/main.py --mode segment
python src/main.py --mode monitor
python src/main.py --mode dashboard
python src/main.py --mode all --small

# ──── Dashboard ────
SKIP_PIPELINE=true docker compose up dashboard   # Dashboard only
DASHBOARD_PORT=9000 docker compose up dashboard   # Custom port
streamlit run src/dashboard/app.py                # Local dev

# ──── Monitoring ────
docker compose ps                      # Service status
docker compose logs -f                 # Follow all logs
docker compose logs -f pipeline        # Follow pipeline logs
docker stats                           # Resource usage

# ──── Debugging ────
docker compose exec pipeline bash      # Shell into pipeline
docker compose exec redis redis-cli    # Redis CLI

# ──── Cleanup ────
docker compose down --rmi local -v     # Full cleanup
docker system prune -f                 # Remove unused Docker resources
```

### Port Reference

| Port | Service | Protocol | Access |
|------|---------|----------|--------|
| 8501 | Streamlit Dashboard | HTTP | `http://localhost:8501` |
| 5001 | MLflow Tracking Server | HTTP | `http://localhost:5001` |
| 6379 | Redis | TCP | `localhost:6379` |

### Configuration File Reference

| File | Description | Key Parameters |
|------|-------------|---------------|
| `config/simulator_config.yaml` | Simulation, personas, churn definition | `num_customers`, `simulation_months`, `churn_definition` |
| `docker-compose.yml` | Container orchestration | Service definitions, ports, volumes |
| `.env` | Environment overrides | `PIPELINE_MODE`, `SMALL`, `BUDGET`, port overrides |
| `Dockerfile.pipeline` | Pipeline image | Python 3.10, build tools, entrypoint |
| `Dockerfile.dashboard` | Dashboard image | Python 3.10, Streamlit, port 8501 |
| `Dockerfile.mlflow` | MLflow image | MLflow 2.12.1, SQLite backend |
| `scripts/pipeline_entrypoint.sh` | Docker pipeline entrypoint | Env-to-CLI flag translation |

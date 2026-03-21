# Deployment Guide

> **Customer Churn Prediction & Retention Optimization System — Configuration and Deployment Reference**

This document covers Docker Compose configuration, environment variables, service
dependencies, scaling considerations, monitoring setup, and operational procedures
for the e-commerce churn prediction system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Docker Compose Architecture](#2-docker-compose-architecture)
   - 2.1 [Service Overview](#21-service-overview)
   - 2.2 [Network Topology](#22-network-topology)
   - 2.3 [Volume Mounts](#23-volume-mounts)
3. [Docker Compose Configuration](#3-docker-compose-configuration)
   - 3.1 [docker-compose.yml Reference](#31-docker-composeyml-reference)
   - 3.2 [Dockerfile.pipeline](#32-dockerfilepipeline)
   - 3.3 [Dockerfile.dashboard](#33-dockerfiledashboard)
4. [Environment Variables](#4-environment-variables)
   - 4.1 [Global Variables](#41-global-variables)
   - 4.2 [Pipeline Container](#42-pipeline-container)
   - 4.3 [Dashboard Container](#43-dashboard-container)
   - 4.4 [Redis Container](#44-redis-container)
   - 4.5 [MLflow Container](#45-mlflow-container)
5. [Service Dependencies](#5-service-dependencies)
   - 5.1 [Startup Order](#51-startup-order)
   - 5.2 [Health Checks](#52-health-checks)
   - 5.3 [Dependency Graph](#53-dependency-graph)
6. [Deployment Procedures](#6-deployment-procedures)
   - 6.1 [First-Time Deployment](#61-first-time-deployment)
   - 6.2 [Pipeline Execution Modes](#62-pipeline-execution-modes)
   - 6.3 [Restarting After Failure](#63-restarting-after-failure)
   - 6.4 [Updating Configuration](#64-updating-configuration)
7. [Scaling Considerations](#7-scaling-considerations)
   - 7.1 [Horizontal Scaling](#71-horizontal-scaling)
   - 7.2 [Vertical Scaling (Resource Limits)](#72-vertical-scaling-resource-limits)
   - 7.3 [Data Volume Scaling](#73-data-volume-scaling)
   - 7.4 [Redis Scaling](#74-redis-scaling)
   - 7.5 [Production Migration Path](#75-production-migration-path)
8. [Monitoring Setup](#8-monitoring-setup)
   - 8.1 [Pipeline State Monitoring](#81-pipeline-state-monitoring)
   - 8.2 [Container Health Monitoring](#82-container-health-monitoring)
   - 8.3 [Model Performance Monitoring](#83-model-performance-monitoring)
   - 8.4 [Data Drift Detection](#84-data-drift-detection)
   - 8.5 [Log Aggregation](#85-log-aggregation)
   - 8.6 [Alert Configuration](#86-alert-configuration)
9. [Troubleshooting](#9-troubleshooting)
10. [Security Considerations](#10-security-considerations)

---

## 1. Prerequisites

| Requirement | Minimum Version | Notes |
|------------|----------------|-------|
| **Docker** | 20.10+ | `docker --version` to verify |
| **Docker Compose** | 2.0+ (V2) | `docker compose version` to verify |
| **Disk Space** | 10 GB free | For images, data, models, and MLflow artifacts |
| **RAM** | 8 GB recommended | Pipeline container is memory-intensive during training |
| **CPU** | 4 cores recommended | All models run on CPU only (no GPU required) |
| **OS** | Linux, macOS, Windows (WSL2) | Docker Desktop on macOS/Windows |

Verify your Docker installation:

```bash
docker --version          # Docker >= 20.10
docker compose version    # Compose >= 2.0
docker info               # Confirm daemon is running
```

---

## 2. Docker Compose Architecture

### 2.1 Service Overview

The system comprises **4 containers** orchestrated by Docker Compose:

| Service | Image Base | Exposed Port | Purpose |
|---------|-----------|-------------|---------|
| `pipeline` | `python:3.10-slim` | 8000 (API) | ML/DL training, data simulation, optimization, real-time scoring API |
| `dashboard` | `python:3.10-slim` | 8501 | Streamlit interactive dashboard |
| `redis` | `redis:7-alpine` | 6379 | Real-time feature store, event streaming, score caching |
| `mlflow` | `python:3.10-slim` | 5000 | Experiment tracking server (SQLite backend) |

### 2.2 Network Topology

```
                    Host Machine
                    ============
                         |
          +--------------+--------------+
          |              |              |
     localhost:8501 localhost:5000 localhost:8000
          |              |              |
  ┌───────┴──────┐ ┌────┴─────┐ ┌──────┴──────┐
  │  dashboard   │ │  mlflow  │ │   pipeline  │
  │  (Streamlit) │ │ (Tracking│ │  (ML/DL +   │
  │              │ │  Server) │ │   API)      │
  └──────┬───────┘ └────┬─────┘ └──────┬──────┘
         |              |              |
         +--------------+--------------+
                        |
              Docker Internal Network
              (churn_prediction_net)
                        |
                 ┌──────┴──────┐
                 │    redis    │
                 │  (port 6379)│
                 └─────────────┘
```

All containers communicate over the Docker bridge network `churn_prediction_net`.
Services reference each other by container name (e.g., `redis:6379`, `mlflow:5000`).

### 2.3 Volume Mounts

Shared volumes ensure data persistence across container restarts:

| Host Path | Container Path | Used By | Purpose |
|-----------|---------------|---------|---------|
| `./config/` | `/app/config/` | pipeline, dashboard | YAML configuration files |
| `./data/` | `/app/data/` | pipeline, dashboard | Raw data and feature store |
| `./models/` | `/app/models/` | pipeline, dashboard | Trained model artifacts |
| `./results/` | `/app/results/` | pipeline, dashboard | Output CSVs, plots, reports |
| `./mlruns/` | `/app/mlruns/` | pipeline, mlflow | MLflow experiment tracking data |
| `./pipeline_state.json` | `/app/pipeline_state.json` | pipeline, dashboard | Pipeline checkpoint state |

> **Important:** All volume mounts are bind mounts (not Docker volumes), so data
> is directly accessible on the host filesystem for inspection and debugging.

---

## 3. Docker Compose Configuration

### 3.1 docker-compose.yml Reference

```yaml
version: "3.8"

services:
  # ─────────────────────────────────────────────────
  # Redis — Real-time feature store & event streaming
  # ─────────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    container_name: churn-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 5s
    restart: unless-stopped
    networks:
      - churn_prediction_net

  # ─────────────────────────────────────────────────
  # MLflow — Experiment tracking server
  # ─────────────────────────────────────────────────
  mlflow:
    build:
      context: .
      dockerfile: Dockerfile.mlflow
    container_name: churn-mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///app/mlruns/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns/artifacts
    command: >
      mlflow server
        --backend-store-uri sqlite:///app/mlruns/mlflow.db
        --default-artifact-root /app/mlruns/artifacts
        --host 0.0.0.0
        --port 5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    restart: unless-stopped
    networks:
      - churn_prediction_net

  # ─────────────────────────────────────────────────
  # Pipeline — ML/DL training & real-time scoring API
  # ─────────────────────────────────────────────────
  pipeline:
    build:
      context: .
      dockerfile: Dockerfile.pipeline
    container_name: churn-pipeline
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./mlruns:/app/mlruns
      - ./pipeline_state.json:/app/pipeline_state.json
    environment:
      - PYTHONPATH=/app
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - PIPELINE_MODE=${PIPELINE_MODE:-train}
      - RANDOM_SEED=${RANDOM_SEED:-42}
      - NUM_CUSTOMERS=${NUM_CUSTOMERS:-20000}
      - BUDGET_KRW=${BUDGET_KRW:-50000000}
    depends_on:
      redis:
        condition: service_healthy
      mlflow:
        condition: service_healthy
    restart: on-failure:3
    networks:
      - churn_prediction_net

  # ─────────────────────────────────────────────────
  # Dashboard — Streamlit interactive UI
  # ─────────────────────────────────────────────────
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: churn-dashboard
    ports:
      - "8501:8501"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./models:/app/models
      - ./results:/app/results
      - ./pipeline_state.json:/app/pipeline_state.json
    environment:
      - PYTHONPATH=/app
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - API_BASE_URL=http://pipeline:8000
    depends_on:
      redis:
        condition: service_healthy
      pipeline:
        condition: service_started
    restart: unless-stopped
    networks:
      - churn_prediction_net

networks:
  churn_prediction_net:
    driver: bridge

volumes:
  redis_data:
```

### 3.2 Dockerfile.pipeline

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ /app/src/
COPY config/ /app/config/

# Default command: run the full pipeline
CMD ["python", "src/main.py", "--mode", "train"]
```

### 3.3 Dockerfile.dashboard

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ /app/dashboard/
COPY src/ /app/src/
COPY config/ /app/config/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

## 4. Environment Variables

### 4.1 Global Variables

These variables apply across all containers:

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Python module search path |
| `TZ` | `Asia/Seoul` | Timezone for log timestamps |

### 4.2 Pipeline Container

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_MODE` | `train` | Execution mode: `train`, `uplift`, `optimize`, `all` |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |
| `NUM_CUSTOMERS` | `20000` | Number of customers to simulate |
| `BUDGET_KRW` | `50000000` | Retention budget in KRW |
| `REDIS_HOST` | `redis` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow tracking server URL |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server bind port |
| `API_KEY` | `churn-api-dev-key-2024` | API authentication key |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SMALL_MODE` | `false` | Use small dataset (5K customers, 6 months) for testing |

### 4.3 Dashboard Container

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://pipeline:8000` | Pipeline API base URL for internal calls |
| `REDIS_HOST` | `redis` | Redis hostname for real-time data |
| `REDIS_PORT` | `6379` | Redis port |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server for experiment browsing |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit port |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Run Streamlit without browser auto-open |

### 4.4 Redis Container

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_MAXMEMORY` | `512mb` | Maximum memory allocation |
| `REDIS_MAXMEMORY_POLICY` | `allkeys-lru` | Eviction policy when memory limit is reached |
| `REDIS_APPENDONLY` | `yes` | Enable AOF persistence |

Redis is configured via command-line flags in `docker-compose.yml`. For advanced
configuration, mount a custom `redis.conf`:

```yaml
redis:
  volumes:
    - ./config/redis.conf:/usr/local/etc/redis/redis.conf
  command: redis-server /usr/local/etc/redis/redis.conf
```

### 4.5 MLflow Container

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_BACKEND_STORE_URI` | `sqlite:///app/mlruns/mlflow.db` | Backend store (SQLite) |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | `/app/mlruns/artifacts` | Artifact storage path |
| `MLFLOW_HOST` | `0.0.0.0` | Server bind address |
| `MLFLOW_PORT` | `5000` | Server bind port |

> **Note:** This deployment uses local SQLite for the backend store and a local
> filesystem for artifacts. No PostgreSQL or S3 is required.

### Override via .env File

Create a `.env` file in the project root to override defaults without modifying
`docker-compose.yml`:

```bash
# .env
PIPELINE_MODE=all
RANDOM_SEED=42
NUM_CUSTOMERS=20000
BUDGET_KRW=50000000
SMALL_MODE=false
LOG_LEVEL=INFO
API_KEY=my-production-api-key
```

Docker Compose automatically loads this file. You can also specify a custom env file:

```bash
docker compose --env-file .env.production up --build
```

---

## 5. Service Dependencies

### 5.1 Startup Order

Services start in a strict dependency order to ensure infrastructure is ready
before application containers launch:

```
1. redis        (no dependencies — starts first)
2. mlflow       (no dependencies — starts in parallel with redis)
3. pipeline     (depends_on: redis [healthy], mlflow [healthy])
4. dashboard    (depends_on: redis [healthy], pipeline [started])
```

The `depends_on` conditions ensure:
- **Redis** is accepting connections (health check passes) before the pipeline starts
- **MLflow** is serving HTTP requests before the pipeline begins experiment logging
- **Pipeline** has started (but not necessarily completed training) before the dashboard launches

### 5.2 Health Checks

Each service defines health checks to validate readiness:

| Service | Health Check | Interval | Timeout | Retries | Start Period |
|---------|-------------|----------|---------|---------|-------------|
| `redis` | `redis-cli ping` | 10s | 5s | 5 | 5s |
| `mlflow` | `curl -f http://localhost:5000/health` | 30s | 10s | 3 | 15s |
| `pipeline` | `curl -f http://localhost:8000/api/v1/health` | 30s | 10s | 5 | 60s |
| `dashboard` | `curl -f http://localhost:8501/_stcore/health` | 30s | 10s | 3 | 30s |

### 5.3 Dependency Graph

```
               ┌──────────┐
               │  redis   │  (infrastructure)
               └────┬─────┘
                    │ healthy
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           │
  ┌──────────┐ ┌──────────┐    │
  │  mlflow  │ │          │    │
  └────┬─────┘ │          │    │
       │healthy│          │    │
       ▼       │          │    │
  ┌──────────┐ │          │    │
  │ pipeline ├─┘          │    │
  └────┬─────┘            │    │
       │ started          │    │
       ▼                  │    │
  ┌──────────┐            │    │
  │dashboard ├────────────┘    │
  └──────────┘                 │
                               │
  (dashboard also depends      │
   on redis being healthy) ────┘
```

### Failure Scenarios

| Scenario | Behavior |
|----------|----------|
| Redis fails to start | Pipeline and dashboard will not start; Docker retries redis up to 5 times |
| MLflow fails to start | Pipeline waits; training can proceed without MLflow but experiment tracking will be unavailable |
| Pipeline crashes during training | `restart: on-failure:3` retries up to 3 times; `pipeline_state.json` enables resumption from last checkpoint |
| Dashboard crashes | `restart: unless-stopped` auto-restarts; dashboard reads from files/Redis so state is preserved |

---

## 6. Deployment Procedures

### 6.1 First-Time Deployment

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/capstone.git
cd capstone

# 2. (Optional) Configure parameters
#    Edit config/*.yaml files or create a .env file
cp .env.example .env
vim .env

# 3. Create required directories
mkdir -p data/raw data/features models results mlruns

# 4. Build and start all services
docker compose up --build

# 5. Monitor pipeline progress
docker compose logs -f pipeline

# 6. Access services once pipeline completes
#    Dashboard:  http://localhost:8501
#    MLflow UI:  http://localhost:5000
#    API Docs:   http://localhost:8000/docs
```

### 6.2 Pipeline Execution Modes

Run specific pipeline modes via environment variable override:

```bash
# Full training pipeline (default)
docker compose up --build

# Run only uplift modeling (assumes training is complete)
PIPELINE_MODE=uplift docker compose up pipeline

# Run budget optimization with custom budget
PIPELINE_MODE=optimize BUDGET_KRW=100000000 docker compose up pipeline

# Run all modes sequentially
PIPELINE_MODE=all docker compose up pipeline

# Quick test with small dataset
SMALL_MODE=true NUM_CUSTOMERS=5000 docker compose up --build
```

Or override the command directly:

```bash
docker compose run pipeline python src/main.py --mode train
docker compose run pipeline python src/main.py --mode uplift
docker compose run pipeline python src/main.py --mode optimize --budget 50000000
```

### 6.3 Restarting After Failure

The pipeline uses checkpoint-based recovery via `pipeline_state.json`:

```bash
# 1. Check pipeline state
cat pipeline_state.json

# 2. Identify the failed step and check logs
docker compose logs pipeline | tail -100

# 3. Fix the issue (e.g., configuration change, resource limit)

# 4. Restart — completed steps are automatically skipped
docker compose up pipeline
```

**Force re-run from scratch** (clears all checkpoints):

```bash
# Reset pipeline state
echo '{}' > pipeline_state.json

# Clear generated data
rm -rf data/raw/* data/features/* models/* results/*

# Rebuild and restart
docker compose up --build
```

### 6.4 Updating Configuration

YAML configuration files are bind-mounted, so changes take effect on container restart:

```bash
# 1. Edit configuration
vim config/simulator_config.yaml

# 2. Restart only the affected service
docker compose restart pipeline

# 3. Or restart with a fresh pipeline state
echo '{}' > pipeline_state.json
docker compose restart pipeline
```

---

## 7. Scaling Considerations

### 7.1 Horizontal Scaling

The current architecture is designed for **single-node deployment** (development
and capstone evaluation). For production horizontal scaling:

| Component | Scaling Strategy |
|-----------|-----------------|
| **Dashboard** | Run multiple Streamlit replicas behind a load balancer (nginx/traefik). Each instance is stateless and reads from shared volumes. |
| **API (Scoring)** | Extract the FastAPI scoring service from the pipeline container into a separate, independently scalable service with Gunicorn + Uvicorn workers. |
| **Redis** | Use Redis Cluster or Redis Sentinel for high availability. Current single-node Redis is sufficient for < 100K customers. |
| **MLflow** | Migrate backend store from SQLite to PostgreSQL. Use S3-compatible storage for artifacts. |
| **Pipeline** | Pipeline is a batch job — scale vertically (more CPU/RAM) rather than horizontally. For distributed training, integrate with Ray or Dask. |

**Example: Scaling the dashboard with replicas:**

```yaml
# docker-compose.override.yml
services:
  dashboard:
    deploy:
      replicas: 3
    ports: []  # Remove host port mapping; use load balancer instead

  nginx:
    image: nginx:alpine
    ports:
      - "8501:80"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - dashboard
```

### 7.2 Vertical Scaling (Resource Limits)

Configure resource limits to prevent containers from consuming all host resources:

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

**Recommended resource allocation by dataset size:**

| Dataset Size | Pipeline RAM | Pipeline CPU | Total System RAM |
|-------------|-------------|-------------|------------------|
| 5K customers (small mode) | 2 GB | 2 cores | 6 GB |
| 20K customers (default) | 4-6 GB | 4 cores | 10 GB |
| 100K customers | 16 GB | 8 cores | 24 GB |
| 500K+ customers | 32 GB+ | 16 cores | 48 GB+ |

### 7.3 Data Volume Scaling

| Data Component | 20K Customers | 100K Customers | 500K Customers |
|---------------|--------------|----------------|----------------|
| Raw event logs | ~200 MB | ~1 GB | ~5 GB |
| Feature store (Parquet) | ~50 MB | ~250 MB | ~1.2 GB |
| Trained models | ~100 MB | ~150 MB | ~200 MB |
| MLflow artifacts | ~200 MB | ~500 MB | ~1 GB |
| **Total disk** | **~600 MB** | **~2 GB** | **~8 GB** |

For large datasets (> 100K customers), consider:
- Switching feature store from Parquet files to a columnar database (DuckDB, ClickHouse)
- Using chunked processing in the feature engineering pipeline
- Enabling Redis persistence with RDB snapshots instead of AOF

### 7.4 Redis Scaling

Current Redis configuration is optimized for the default 20K customer dataset:

```
maxmemory: 512mb
maxmemory-policy: allkeys-lru
```

**Scaling Redis memory by dataset size:**

| Customers | Recommended maxmemory | Key Count (approx.) |
|-----------|----------------------|---------------------|
| 5K | 128mb | ~15K keys |
| 20K | 512mb | ~60K keys |
| 100K | 2gb | ~300K keys |
| 500K | 8gb | ~1.5M keys |

Redis keys include:
- `score:{customer_id}` — cached churn scores (TTL: 300s)
- `features:{customer_id}` — cached feature vectors (TTL: 600s)
- `customer_events` — Redis Stream for real-time event ingestion

### 7.5 Production Migration Path

For moving from development to production:

| Component | Development | Production |
|-----------|-------------|------------|
| **MLflow Backend** | SQLite (`sqlite:///mlruns/mlflow.db`) | PostgreSQL (`postgresql://...`) |
| **MLflow Artifacts** | Local filesystem (`/app/mlruns/artifacts`) | S3 / MinIO (`s3://bucket/artifacts`) |
| **Redis** | Single node, no auth | Redis Sentinel/Cluster with AUTH |
| **API Auth** | Static API key | OAuth2 / JWT tokens |
| **Secrets** | `.env` file | Docker Secrets / Vault |
| **Logging** | stdout/stderr | ELK Stack / CloudWatch |
| **Monitoring** | File-based (`monitoring_report.json`) | Prometheus + Grafana |
| **Orchestration** | Docker Compose | Kubernetes (Helm charts) |
| **CI/CD** | Manual `docker compose up` | GitHub Actions + ArgoCD |

---

## 8. Monitoring Setup

### 8.1 Pipeline State Monitoring

The pipeline tracks execution state in `pipeline_state.json`:

```json
{
  "simulation": "completed",
  "feature_engineering": "completed",
  "ml_training": "completed",
  "dl_training": "completed",
  "ensemble": "completed",
  "uplift_modeling": "completed",
  "clv_prediction": "completed",
  "segmentation": "completed",
  "budget_optimization": "completed",
  "ab_testing": "completed",
  "survival_analysis": "completed",
  "monitoring": "completed",
  "recommendations": "completed",
  "timestamp": "2026-03-21T10:00:00",
  "total_duration_seconds": 1847
}
```

Each step has one of three states:
- `completed` — step finished successfully
- `failed` — step encountered an error (details in logs)
- `pending` — step has not yet been executed

**Check pipeline state:**

```bash
# From host
cat pipeline_state.json | python -m json.tool

# From within Docker
docker compose exec pipeline cat /app/pipeline_state.json
```

### 8.2 Container Health Monitoring

Monitor container status and resource usage:

```bash
# Container status (includes health check results)
docker compose ps

# Real-time resource usage
docker stats churn-pipeline churn-dashboard churn-redis churn-mlflow

# Individual container health
docker inspect --format='{{.State.Health.Status}}' churn-redis
docker inspect --format='{{.State.Health.Status}}' churn-mlflow

# Service logs (all services)
docker compose logs -f --tail=100

# Service logs (specific service)
docker compose logs -f pipeline
docker compose logs -f dashboard
```

**Expected healthy output from `docker compose ps`:**

```
NAME              COMMAND                 STATUS                    PORTS
churn-redis       "redis-server ..."      Up 2 hours (healthy)      0.0.0.0:6379->6379/tcp
churn-mlflow      "mlflow server ..."     Up 2 hours (healthy)      0.0.0.0:5000->5000/tcp
churn-pipeline    "python src/main.py"    Up 2 hours                0.0.0.0:8000->8000/tcp
churn-dashboard   "streamlit run ..."     Up 2 hours                0.0.0.0:8501->8501/tcp
```

### 8.3 Model Performance Monitoring

Model metrics are tracked in two places:

**1. MLflow UI (http://localhost:5000):**
- AUC-ROC, Precision, Recall, F1 for ML and DL models
- Ensemble performance metrics
- Hyperparameter values and training parameters
- Model artifacts and version history

**2. Results directory (`results/`):**

| File | Contents |
|------|----------|
| `monitoring_report.json` | PSI/KS drift scores, threshold alerts, performance trends |
| `model_metrics.json` | Latest training/test metrics for all models |
| `shap_summary.png` | SHAP feature importance visualization |

**Query MLflow metrics programmatically:**

```bash
# List experiments
curl -s http://localhost:5000/api/2.0/mlflow/experiments/search | python -m json.tool

# Get run metrics
curl -s "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -d '{"experiment_ids": ["1"]}' | python -m json.tool
```

### 8.4 Data Drift Detection

The monitoring module (`src/monitoring/drift_detector.py`) computes:

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| **PSI** (Population Stability Index) | Measures distribution shift between training and scoring data | PSI > 0.2 (significant drift) |
| **KS-test** (Kolmogorov-Smirnov) | Statistical test for distribution difference per feature | p-value < 0.05 |
| **Feature mean/std shift** | Tracks statistical moments over time | > 2 standard deviations from training baseline |

**Monitoring report structure (`results/monitoring_report.json`):**

```json
{
  "timestamp": "2026-03-21T10:00:00",
  "overall_drift_detected": false,
  "psi_scores": {
    "days_since_last_purchase": 0.08,
    "purchase_frequency_change_4w": 0.12,
    "session_duration_trend": 0.05
  },
  "ks_test_results": {
    "days_since_last_purchase": {"statistic": 0.04, "p_value": 0.32},
    "purchase_frequency_change_4w": {"statistic": 0.07, "p_value": 0.08}
  },
  "alerts": [],
  "model_performance": {
    "auc_roc": 0.861,
    "auc_roc_baseline": 0.858,
    "performance_degradation": false
  }
}
```

### 8.5 Log Aggregation

All services write logs to stdout/stderr, collected by Docker's logging driver.

**Configure structured JSON logging:**

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

**View logs with timestamps:**

```bash
# All services with timestamps
docker compose logs -f --timestamps

# Pipeline only, last 200 lines
docker compose logs -f --tail=200 pipeline

# Export logs to file
docker compose logs pipeline > logs/pipeline_$(date +%Y%m%d).log
```

**Application log levels** (configurable via `LOG_LEVEL` env var):

| Level | Description |
|-------|-------------|
| `DEBUG` | Detailed execution trace (feature engineering steps, model layer outputs) |
| `INFO` | Pipeline progress, metrics, completion messages (default) |
| `WARNING` | Non-fatal issues (missing optional features, fallback behavior) |
| `ERROR` | Step failures, exceptions, data validation errors |

### 8.6 Alert Configuration

For automated alerting, integrate with external monitoring tools:

**Option A: Simple file-based alerting (built-in)**

The pipeline writes alerts to `results/alerts.json` when drift or performance
degradation is detected. The dashboard displays these alerts on the Monitoring tab.

**Option B: Webhook-based alerting (custom)**

Configure alert webhooks in `config/monitoring_config.yaml`:

```yaml
monitoring:
  alerts:
    enabled: true
    check_interval_hours: 24
    channels:
      - type: slack
        webhook_url: "https://hooks.slack.com/services/..."
        events: ["drift_detected", "model_degradation", "pipeline_failure"]
      - type: email
        smtp_host: "smtp.gmail.com"
        smtp_port: 587
        recipients: ["team@example.com"]
        events: ["pipeline_failure"]
    thresholds:
      psi_warning: 0.1
      psi_critical: 0.2
      auc_drop_warning: 0.02
      auc_drop_critical: 0.05
```

---

## 9. Troubleshooting

### Common Issues

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `redis` container exits immediately | Port 6379 already in use | `lsof -i :6379` and stop conflicting process, or change port in compose |
| Pipeline OOM killed | Insufficient memory for training | Increase Docker memory limit or use `SMALL_MODE=true` |
| MLflow UI shows no experiments | `mlruns/` directory permissions | `chmod -R 777 mlruns/` or run containers as the same UID |
| Dashboard shows "No data available" | Pipeline hasn't completed yet | Check `pipeline_state.json` and wait for completion |
| Pipeline stuck at "pending" step | Previous step failed silently | Check logs: `docker compose logs pipeline` |
| Redis connection refused | Redis not healthy yet | Wait for health check; increase `start_period` |
| Models show poor performance | Wrong random seed or config | Verify `RANDOM_SEED=42` and check `config/` files |
| Permission denied on volume mounts | Docker user/group mismatch | Add `user: "${UID}:${GID}"` to compose service |

### Diagnostic Commands

```bash
# Full system status
docker compose ps -a

# Check container resource usage
docker stats --no-stream

# Inspect a specific container
docker compose exec pipeline bash
docker compose exec redis redis-cli info

# Test Redis connectivity from pipeline
docker compose exec pipeline python -c "import redis; r=redis.Redis('redis'); print(r.ping())"

# Test MLflow connectivity
docker compose exec pipeline python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); print(mlflow.get_tracking_uri())"

# Validate pipeline state
docker compose exec pipeline python -c "import json; print(json.dumps(json.load(open('pipeline_state.json')), indent=2))"

# Check disk usage by service
docker system df -v
```

### Reset and Clean Restart

```bash
# Stop everything
docker compose down

# Remove all generated data
rm -rf data/raw/* data/features/* models/* results/* mlruns/*
echo '{}' > pipeline_state.json

# Remove Docker images (force rebuild)
docker compose down --rmi local

# Remove volumes (including Redis data)
docker compose down -v

# Fresh start
docker compose up --build
```

---

## 10. Security Considerations

### Development vs. Production

| Aspect | Development (Current) | Production (Recommended) |
|--------|----------------------|--------------------------|
| API Key | Hardcoded in config (`churn-api-dev-key-2024`) | Rotate via Docker Secrets / env var |
| Redis Auth | No password | Set `requirepass` in redis.conf |
| MLflow Access | Open to all on network | Add authentication proxy (nginx + basic auth) |
| Network | Docker bridge (all services visible) | Isolate services into frontend/backend networks |
| Secrets | `.env` file | Docker Secrets, HashiCorp Vault, or cloud KMS |
| TLS | None | Terminate TLS at load balancer or reverse proxy |
| Container User | Root (default) | Non-root user with minimal privileges |

### Network Isolation (Production)

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
    networks:
      - backend
  mlflow:
    networks:
      - backend
  pipeline:
    networks:
      - frontend   # API exposed to host
      - backend    # Access to redis, mlflow
  dashboard:
    networks:
      - frontend   # Streamlit exposed to host
      - backend    # Access to redis for real-time data
```

### Sensitive Files

Never commit these files to version control:

```gitignore
# .gitignore
.env
.env.production
config/secrets.yaml
*.pem
*.key
```

---

## Appendix A: Quick Reference Commands

```bash
# ──── Lifecycle ────
docker compose up --build              # Build and start all services
docker compose up -d --build           # Detached mode
docker compose down                    # Stop all services
docker compose down -v                 # Stop and remove volumes
docker compose restart pipeline        # Restart specific service

# ──── Monitoring ────
docker compose ps                      # Service status
docker compose logs -f                 # Follow all logs
docker compose logs -f pipeline        # Follow pipeline logs
docker stats                           # Resource usage

# ──── Pipeline ────
PIPELINE_MODE=train docker compose up pipeline
PIPELINE_MODE=uplift docker compose up pipeline
PIPELINE_MODE=optimize docker compose up pipeline
SMALL_MODE=true docker compose up --build

# ──── Debugging ────
docker compose exec pipeline bash      # Shell into pipeline
docker compose exec redis redis-cli    # Redis CLI
cat pipeline_state.json                # Check pipeline progress

# ──── Cleanup ────
docker compose down --rmi local -v     # Full cleanup
docker system prune -f                 # Remove unused Docker resources
```

---

## Appendix B: Port Reference

| Port | Service | Protocol | Access |
|------|---------|----------|--------|
| 8501 | Streamlit Dashboard | HTTP | `http://localhost:8501` |
| 8000 | Pipeline API (FastAPI) | HTTP | `http://localhost:8000` |
| 5000 | MLflow Tracking Server | HTTP | `http://localhost:5000` |
| 6379 | Redis | TCP | `localhost:6379` (or `redis:6379` internally) |

---

## Appendix C: Configuration File Reference

| File | Description | Key Parameters |
|------|-------------|---------------|
| `config/simulator_config.yaml` | Simulation & persona settings | `num_customers`, `simulation_months`, `personas`, `churn_definition` |
| `config/base_config.yaml` | Global paths and seed | `random_seed`, `paths`, `churn_definition` |
| `config/feature_config.yaml` | Feature engineering | Feature list, window sizes, aggregation methods |
| `config/model_config.yaml` | ML/DL hyperparameters | `train_months`, `test_months`, `ensemble_weights`, learning rates |
| `config/uplift_config.yaml` | Uplift modeling | T-Learner/S-Learner settings, quadrant thresholds |
| `config/optimization_config.yaml` | Budget optimization | `total_budget_krw`, what-if scenarios, constraint parameters |
| `config/dashboard_config.yaml` | Dashboard layout | Tab configuration, chart settings, refresh intervals |
| `config/api_config.yaml` | API settings | Auth key, rate limits, cache TTL, risk thresholds |
| `docker-compose.yml` | Container orchestration | Service definitions, ports, volumes, dependencies |
| `.env` | Environment overrides | Runtime variable overrides |

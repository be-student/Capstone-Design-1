#!/bin/bash
# =============================================================================
# MLflow Server Entrypoint Script
# =============================================================================
# Starts the MLflow tracking server with SQLite backend and local artifact store.
# All settings are configurable via environment variables.
# =============================================================================

set -e

# Defaults (can be overridden via environment variables)
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow/mlflow.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

# Ensure artifact directory exists
mkdir -p "${MLFLOW_ARTIFACT_ROOT}"

echo "=========================================="
echo "  MLflow Tracking Server"
echo "=========================================="
echo "  Backend Store:   ${MLFLOW_BACKEND_STORE_URI}"
echo "  Artifact Root:   ${MLFLOW_ARTIFACT_ROOT}"
echo "  Host:            ${MLFLOW_HOST}"
echo "  Port:            ${MLFLOW_PORT}"
echo "=========================================="

# Start MLflow tracking server
exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}" \
    --serve-artifacts

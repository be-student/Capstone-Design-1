#!/usr/bin/env bash
# =============================================================================
# Pipeline Container Entrypoint
# =============================================================================
# Flexible entrypoint for the ML/DL pipeline container.
# Builds the CLI command from environment variables or passes through CLI args.
#
# Environment Variables:
#   PIPELINE_MODE  - Pipeline mode (default: all)
#   SMALL          - "true" for small dataset mode (default: false)
#   BUDGET         - Budget cap for optimize mode (optional)
#   VERBOSE        - "true" for verbose output (default: false)
#   PIPELINE_CONFIG     - Config path passed as --config (optional)
#   PIPELINE_DATA_DIR   - Data directory passed as --data (optional)
#   PIPELINE_OUTPUT_DIR - Output base directory passed as --output (optional)
#   PIPELINE_WAIT_FOR_SERVICES - "false" to skip Docker service waits
#   PIPELINE_WAIT_TIMEOUT      - Seconds to wait for MLflow/Redis (default: 120)
#
# Usage:
#   # Via env vars (docker-compose style):
#   PIPELINE_MODE=train SMALL=true ./pipeline_entrypoint.sh
#
#   # Via CLI args (direct override):
#   ./pipeline_entrypoint.sh --mode train --small
# =============================================================================

set -euo pipefail

# ANSI colours
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[PIPELINE]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_ok()    { echo -e "${GREEN}[PIPELINE]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[PIPELINE]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_error() { echo -e "${RED}[PIPELINE]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"; }

detect_runtime() {
    if [ -z "${PIPELINE_RUNTIME:-}" ]; then
        if [ -f /.dockerenv ]; then
            export PIPELINE_RUNTIME="docker"
        elif [ -r /proc/1/cgroup ] && grep -qaE "docker|containerd|kubepods" /proc/1/cgroup; then
            export PIPELINE_RUNTIME="docker"
        fi
    fi
}

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout="${PIPELINE_WAIT_TIMEOUT:-120}"
    local start
    start="$(date +%s)"

    log_info "Waiting for ${name}: ${url}"
    while true; do
        if python - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        sys.exit(0 if response.status < 500 else 1)
except Exception:
    sys.exit(1)
PY
        then
            log_ok "${name} is reachable"
            return 0
        fi

        if [ $(( $(date +%s) - start )) -ge "${timeout}" ]; then
            log_error "Timed out waiting for ${name}: ${url}"
            return 1
        fi
        sleep 2
    done
}

wait_for_tcp() {
    local name="$1"
    local host="$2"
    local port="$3"
    local timeout="${PIPELINE_WAIT_TIMEOUT:-120}"
    local start
    start="$(date +%s)"

    log_info "Waiting for ${name}: ${host}:${port}"
    while true; do
        if python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=3):
        sys.exit(0)
except Exception:
    sys.exit(1)
PY
        then
            log_ok "${name} is reachable"
            return 0
        fi

        if [ $(( $(date +%s) - start )) -ge "${timeout}" ]; then
            log_error "Timed out waiting for ${name}: ${host}:${port}"
            return 1
        fi
        sleep 2
    done
}

prepare_runtime() {
    detect_runtime

    if [ "${PIPELINE_RUNTIME:-}" = "docker" ] \
        && { [ -n "${MLFLOW_TRACKING_URI:-}" ] || [ -n "${MLFLOW_ARTIFACT_ROOT:-}" ]; }; then
        export MLFLOW_ARTIFACT_LOCATION="${MLFLOW_ARTIFACT_LOCATION:-${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}}"
    fi

    if [ "${PIPELINE_WAIT_FOR_SERVICES:-true}" = "false" ]; then
        return 0
    fi

    if [[ "${MLFLOW_TRACKING_URI:-}" =~ ^https?:// ]]; then
        wait_for_http "MLflow" "${MLFLOW_TRACKING_URI%/}/health"
    fi

    if [ -n "${REDIS_HOST:-}" ]; then
        wait_for_tcp "Redis" "${REDIS_HOST}" "${REDIS_PORT:-6379}"
    fi
}

prepare_runtime

# ---------------------------------------------------------------------------
# If CLI args are provided, pass them directly to src.main
# ---------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    log_info "Running with CLI args: $*"
    exec python -m src.main "$@"
fi

# ---------------------------------------------------------------------------
# Otherwise, build command from environment variables
# ---------------------------------------------------------------------------
PIPELINE_MODE="${PIPELINE_MODE:-all}"
SMALL="${SMALL:-false}"
BUDGET="${BUDGET:-}"
VERBOSE="${VERBOSE:-false}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-}"
PIPELINE_DATA_DIR="${PIPELINE_DATA_DIR:-}"
PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-}"

log_info "=========================================="
log_info "  Churn Prediction Pipeline"
log_info "=========================================="
log_info "  Mode:    ${PIPELINE_MODE}"
log_info "  Small:   ${SMALL}"
log_info "  Budget:  ${BUDGET:-<default>}"
log_info "  Verbose: ${VERBOSE}"
log_info "  Runtime: ${PIPELINE_RUNTIME:-local}"
log_info "  MLflow:  ${MLFLOW_TRACKING_URI:-<config default>}"
log_info "  Redis:   ${REDIS_HOST:-<config default>}:${REDIS_PORT:-6379}"
log_info "=========================================="

# Build argument list
ARGS=(--mode "${PIPELINE_MODE}")

if [ -n "${PIPELINE_CONFIG}" ]; then
    ARGS+=(--config "${PIPELINE_CONFIG}")
fi

if [ -n "${PIPELINE_DATA_DIR}" ]; then
    ARGS+=(--data "${PIPELINE_DATA_DIR}")
fi

if [ -n "${PIPELINE_OUTPUT_DIR}" ]; then
    ARGS+=(--output "${PIPELINE_OUTPUT_DIR}")
fi

if [ "${SMALL}" = "true" ]; then
    ARGS+=(--small)
fi

if [ -n "${BUDGET}" ]; then
    ARGS+=(--budget "${BUDGET}")
fi

if [ "${VERBOSE}" = "true" ]; then
    ARGS+=(-v)
fi

log_info "Executing: python -m src.main ${ARGS[*]}"

# Execute pipeline
exec python -m src.main "${ARGS[@]}"

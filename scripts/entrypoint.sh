#!/usr/bin/env bash
# =============================================================================
# Entrypoint / Orchestration Script
# =============================================================================
# Runs the full ML pipeline (data generation, model training, evaluation)
# to completion, then launches the Streamlit dashboard.
#
# Usage:
#   ./scripts/entrypoint.sh                    # Full pipeline + dashboard
#   ./scripts/entrypoint.sh --skip-pipeline    # Dashboard only (data exists)
#   ./scripts/entrypoint.sh --pipeline-only    # Pipeline only (no dashboard)
#   ./scripts/entrypoint.sh --small            # Small mode for testing
#
# Environment Variables:
#   SKIP_PIPELINE       - Set to "true" to skip pipeline execution
#   PIPELINE_ONLY       - Set to "true" to skip dashboard launch
#   PIPELINE_ARGS       - Extra args passed to pipeline (e.g., "--small -v")
#   STREAMLIT_PORT      - Dashboard port (default: 8501)
#   PIPELINE_STATE_FILE - Path to pipeline state JSON (default: data/pipeline_state.json)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration with sensible defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
STREAMLIT_HOST="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
PIPELINE_STATE_FILE="${PIPELINE_STATE_FILE:-${PROJECT_ROOT}/data/pipeline_state.json}"
PIPELINE_ARGS="${PIPELINE_ARGS:-}"
SKIP_PIPELINE="${SKIP_PIPELINE:-false}"
PIPELINE_ONLY="${PIPELINE_ONLY:-false}"
MAX_RETRIES="${PIPELINE_MAX_RETRIES:-1}"

# ANSI colours for log output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log_info()  { echo -e "${BLUE}[INFO]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*"; }

# ---------------------------------------------------------------------------
# Parse CLI flags (override env vars)
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --skip-pipeline)  SKIP_PIPELINE="true" ;;
        --pipeline-only)  PIPELINE_ONLY="true" ;;
        --small)          PIPELINE_ARGS="${PIPELINE_ARGS} --small" ;;
        -v|--verbose)     PIPELINE_ARGS="${PIPELINE_ARGS} -v" ;;
        -q|--quiet)       PIPELINE_ARGS="${PIPELINE_ARGS} -q" ;;
        *)                PIPELINE_ARGS="${PIPELINE_ARGS} $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Run the ML Pipeline
# ---------------------------------------------------------------------------
run_pipeline() {
    log_info "=========================================="
    log_info "  Starting ML Pipeline"
    log_info "=========================================="

    local attempt=1
    local pipeline_exit_code=0

    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        log_info "Pipeline attempt ${attempt}/${MAX_RETRIES}"

        # shellcheck disable=SC2086
        python -m src.main --mode all ${PIPELINE_ARGS} && pipeline_exit_code=0 || pipeline_exit_code=$?

        if [ "$pipeline_exit_code" -eq 0 ]; then
            log_ok "Pipeline completed successfully (attempt ${attempt})"
            return 0
        fi

        log_warn "Pipeline failed with exit code ${pipeline_exit_code} (attempt ${attempt}/${MAX_RETRIES})"

        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            log_info "Retrying in 5 seconds (checkpoint/resume enabled)..."
            sleep 5
        fi

        attempt=$((attempt + 1))
    done

    log_error "Pipeline failed after ${MAX_RETRIES} attempt(s)"
    return "$pipeline_exit_code"
}

# ---------------------------------------------------------------------------
# Step 2: Launch the Streamlit Dashboard
# ---------------------------------------------------------------------------
launch_dashboard() {
    log_info "=========================================="
    log_info "  Launching Streamlit Dashboard"
    log_info "  Port: ${STREAMLIT_PORT}"
    log_info "  Host: ${STREAMLIT_HOST}"
    log_info "=========================================="

    exec streamlit run src/dashboard/app.py \
        --server.port="${STREAMLIT_PORT}" \
        --server.address="${STREAMLIT_HOST}" \
        --server.headless=true
}

# ---------------------------------------------------------------------------
# Step 3: Verify pipeline outputs exist
# ---------------------------------------------------------------------------
verify_pipeline_outputs() {
    local data_dir="${PROJECT_ROOT}/data/raw"
    local results_dir="${PROJECT_ROOT}/results"
    local models_dir="${PROJECT_ROOT}/models"
    local missing=0

    log_info "Verifying pipeline outputs..."

    # Check for generated data
    if [ -d "$data_dir" ] && [ "$(ls -A "$data_dir" 2>/dev/null)" ]; then
        log_ok "Data directory has files: ${data_dir}"
    else
        log_warn "Data directory is empty or missing: ${data_dir}"
        missing=$((missing + 1))
    fi

    # Check for model artifacts
    if [ -d "$models_dir" ] && [ "$(ls -A "$models_dir" 2>/dev/null)" ]; then
        log_ok "Models directory has files: ${models_dir}"
    else
        log_warn "Models directory is empty or missing: ${models_dir}"
        missing=$((missing + 1))
    fi

    # Check for results
    if [ -d "$results_dir" ] && [ "$(ls -A "$results_dir" 2>/dev/null)" ]; then
        log_ok "Results directory has files: ${results_dir}"
    else
        log_warn "Results directory is empty or missing: ${results_dir}"
        missing=$((missing + 1))
    fi

    if [ "$missing" -gt 0 ]; then
        log_warn "${missing} output directories are empty/missing"
        return 1
    fi

    log_ok "All pipeline outputs verified"
    return 0
}

# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
main() {
    log_info "=========================================="
    log_info "  Churn Prediction System - Entrypoint"
    log_info "=========================================="
    log_info "  SKIP_PIPELINE:  ${SKIP_PIPELINE}"
    log_info "  PIPELINE_ONLY:  ${PIPELINE_ONLY}"
    log_info "  PIPELINE_ARGS:  ${PIPELINE_ARGS:-<none>}"
    log_info "=========================================="

    cd "${PROJECT_ROOT}"

    # Ensure output directories exist
    mkdir -p data/raw results models

    # --- Pipeline Phase ---
    if [ "${SKIP_PIPELINE}" = "true" ]; then
        log_info "Skipping pipeline (SKIP_PIPELINE=true)"

        if ! verify_pipeline_outputs; then
            log_warn "Pipeline outputs missing but SKIP_PIPELINE is set. Dashboard may show empty data."
        fi
    else
        if ! run_pipeline; then
            log_error "Pipeline execution failed. Aborting."
            exit 1
        fi
        verify_pipeline_outputs || true
    fi

    # --- Dashboard Phase ---
    if [ "${PIPELINE_ONLY}" = "true" ]; then
        log_info "Pipeline-only mode. Skipping dashboard launch."
        log_ok "All done."
        exit 0
    fi

    launch_dashboard
}

main

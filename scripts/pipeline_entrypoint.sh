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

log_info "=========================================="
log_info "  Churn Prediction Pipeline"
log_info "=========================================="
log_info "  Mode:    ${PIPELINE_MODE}"
log_info "  Small:   ${SMALL}"
log_info "  Budget:  ${BUDGET:-<default>}"
log_info "  Verbose: ${VERBOSE}"
log_info "=========================================="

# Build argument list
ARGS=(--mode "${PIPELINE_MODE}")

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

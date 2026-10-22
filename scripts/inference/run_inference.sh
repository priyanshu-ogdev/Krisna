#!/usr/bin/env bash
# =============================================================================
# scripts/inference/run_inference.sh
# Inference launcher for Krivex Swap Orchestrator and service stack (Linux / WSL2).
#
# Usage:
#   ./scripts/inference/run_inference.sh                    # MockBackend, no GPU needed
#   KRIVEX_USE_REAL_BACKENDS=1 ./scripts/inference/run_inference.sh
#   KRIVEX_USE_REAL_BACKENDS=1 KRIVEX_LOW_VRAM_MODE=1 ./scripts/inference/run_inference.sh
#   ./scripts/inference/run_inference.sh 0.0.0.0 8420       # bind host/port passthrough
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [ "${KRIVEX_USE_REAL_BACKENDS:-0}" = "1" ] && [ "${KRIVEX_LOW_VRAM_MODE:-0}" = "1" ]; then
    export KRIVEX_RAM_ENVELOPE_GB="${KRIVEX_RAM_ENVELOPE_GB:-128.0}"
fi

exec "${REPO_ROOT}/scripts/inference/run_service.sh" "$@"

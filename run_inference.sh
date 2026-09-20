#!/usr/bin/env bash
# Root-level inference launcher. Thin wrapper around
# scripts/inference/run_service.sh — kept there as the source of truth
# (it has the real host/port arg parsing and startup banner logic); this
# just gives a stable root-level entry point and documents the env vars
# that matter for THIS project's actual target hardware (single RTX A6000,
# 48GB VRAM, 128GB system RAM — see docs/review/18_training_memory_audit.md
# and the earlier VRAM/RAM headroom fixes throughout docs/review/).
#
# Usage:
#   ./run_inference.sh                    # MockBackend, no GPU needed
#   KRISNA_USE_REAL_BACKENDS=1 ./run_inference.sh
#   KRISNA_USE_REAL_BACKENDS=1 KRISNA_LOW_VRAM_MODE=1 ./run_inference.sh
#   ./run_inference.sh 0.0.0.0 8080       # bind host/port passthrough

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "${KRISNA_USE_REAL_BACKENDS:-0}" = "1" ] && [ "${KRISNA_LOW_VRAM_MODE:-0}" = "1" ]; then
    # Defaults in service.py/swap_orchestrator.py assume a more modest
    # RAM budget (KRISNA_RAM_ENVELOPE_GB defaults to 64.0) than this
    # project's actual target machine has (128GB). Not wrong — a
    # conservative default is the right choice for a value nobody has
    # explicitly confirmed for their own hardware — but worth setting
    # explicitly here rather than leaving real headroom unused.
    export KRISNA_RAM_ENVELOPE_GB="${KRISNA_RAM_ENVELOPE_GB:-128.0}"
fi

exec ./scripts/inference/run_service.sh "$@"

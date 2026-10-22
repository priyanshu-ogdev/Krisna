#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/inference/run_service.sh

BIND_HOST="${1:-127.0.0.1}"
PORT="${2:-8420}"

echo -e "\033[1;36mStarting Krivex Inference service on ${BIND_HOST}:${PORT}...\033[0m"
if [ "${KRIVEX_USE_REAL_BACKENDS:-0}" = "1" ]; then
    echo -e "\033[1;33mKRIVEX_USE_REAL_BACKENDS=1 — real model backends, GPU required.\033[0m"
    echo -e "\033[1;33mPlanner tier uses frozen Qwen3.5-9B GGUF; Polish tier uses Qwen-Image-2.1.\033[0m"
    if [ "${KRIVEX_LOW_VRAM_MODE:-0}" = "1" ]; then
        echo -e "\033[1;35mKRIVEX_LOW_VRAM_MODE=1 — targeting a ${KRIVEX_VRAM_ENVELOPE_GB:-12.0}GB GPU envelope,\033[0m"
        echo -e "\033[1;35moffloading to system RAM (peak limit: ${KRIVEX_RAM_ENVELOPE_GB:-16.0}GB).\033[0m"
        echo -e "\033[1;35mResident Planner Review mode operates within unified memory limits.\033[0m"
    fi
else
    echo -e "\033[1;33mMockBackend mode (default) — no GPU/weights needed.\033[0m"
    echo -e "\033[1;33mSet KRIVEX_USE_REAL_BACKENDS=1 to use real model backends.\033[0m"
fi

source .venv/bin/activate
python -m uvicorn krivex_inference.orchestrator.service:app --host "$BIND_HOST" --port "$PORT" --reload

#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/inference/run_service.sh

BIND_HOST="${1:-127.0.0.1}"
PORT="${2:-8420}"

echo -e "\033[1;36mStarting Krisna Inference service on ${BIND_HOST}:${PORT}...\033[0m"
if [ "${KRISNA_USE_REAL_BACKENDS:-0}" = "1" ]; then
    echo -e "\033[1;33mKRISNA_USE_REAL_BACKENDS=1 — real model backends, GPU required.\033[0m"
    echo -e "\033[1;33mCritic tier launches ./venv-critic/bin/python as a subprocess\033[0m"
    echo -e "\033[1;33m(override with KRISNA_CRITIC_VENV_PYTHON). Run\033[0m"
    echo -e "\033[1;33m./scripts/training/setup_env_critic.sh first if you haven't.\033[0m"
    if [ "${KRISNA_LOW_VRAM_MODE:-0}" = "1" ]; then
        echo -e "\033[1;35mKRISNA_LOW_VRAM_MODE=1 — targeting a ${KRISNA_VRAM_ENVELOPE_GB:-12.0}GB GPU envelope,\033[0m"
        echo -e "\033[1;35moffloading the rest to system RAM (up to ${KRISNA_RAM_ENVELOPE_GB:-64.0}GB).\033[0m"
        echo -e "\033[1;35mSee docs/architecture/RESEARCH_AND_CITATIONS.md §4 and\033[0m"
        echo -e "\033[1;35mdocs/review/06_inference_orchestrator.md for the real cost breakdown —\033[0m"
        echo -e "\033[1;35mthe Critic tier's offload alone can need ~40GB system RAM.\033[0m"
    fi
    echo -e "\033[1;36m[+] Verifying GPU and deployment readiness...\033[0m"
    python scripts/inference/check_hardware.py --require-gpu ${KRISNA_LOW_VRAM_MODE:+--low-vram}
else
    echo -e "\033[1;33mMockBackend mode — no GPU/weights needed (test/CI only).\033[0m"
    echo -e "\033[1;33mSet KRISNA_USE_REAL_BACKENDS=1 for production deployment.\033[0m"
fi

source .venv/bin/activate
python -m uvicorn krisna_inference.orchestrator.service:app --host "$BIND_HOST" --port "$PORT" --reload

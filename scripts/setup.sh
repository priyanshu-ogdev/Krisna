#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh
# Monorepo environment setup — orchestrates per-subsystem setup scripts.
#
# Phases (run all by default, or pass specific ones as args):
#   base      - main .venv: krivex-inference[dev] + krivex-training[dev] + krivex-data-forge
#   backends  - installs real inference dependencies (torch, diffusers, transformers, etc.)
#   polish    - installs Polish/DPO training dependencies (diffusers, peft, accelerate)
#   verifiers - CLIP/OCR/aesthetic verifier-stack dependencies
#   frontend  - npm install for inference/frontend/ (Studio web interface)
# =============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then
    PHASES=(base backends polish verifiers)
fi

echo -e "\033[1;35m=========================================================\033[0m"
echo -e "\033[1;35mKrivex monorepo setup — phases: ${PHASES[*]}\033[0m"
echo -e "\033[1;35m=========================================================\033[0m"

for phase in "${PHASES[@]}"; do
    case "$phase" in
        base)
            echo -e "\033[1;36m[base] initializing main .venv...\033[0m"
            if [ ! -d ".venv" ]; then
                python3 -m venv .venv
            fi
            # shellcheck disable=SC1091
            source .venv/bin/activate
            pip install --upgrade pip setuptools wheel
            pip install -e "./inference[dev]" --no-deps
            pip install -e "./training[dev]" --no-deps
            pip install -e "./data-forge" --no-deps
            ;;
        backends)
            echo -e "\033[1;36m[backends] setting up inference backend dependencies...\033[0m"
            "${REPO_ROOT}/scripts/inference/setup_env.sh"
            ;;
        polish)
            echo -e "\033[1;36m[polish] setting up polish & DPO training dependencies...\033[0m"
            "${REPO_ROOT}/scripts/training/prepare_polish_dataset.sh" || true
            ;;
        verifiers)
            echo -e "\033[1;36m[verifiers] setting up verification stack...\033[0m"
            "${REPO_ROOT}/scripts/inference/setup_env_verifiers.sh" || true
            ;;
        frontend)
            echo -e "\033[1;36m[frontend] setting up Node.js frontend studio...\033[0m"
            if command -v npm &>/dev/null; then
                (cd "${REPO_ROOT}/inference/frontend" && npm install)
            else
                echo -e "\033[1;31m[frontend] npm not found — please install Node.js 18+.\033[0m"
            fi
            ;;
        *)
            echo -e "\033[1;31mUnknown setup phase: $phase\033[0m"
            exit 1
            ;;
    esac
done

echo -e "\033[1;32m=========================================================\033[0m"
echo -e "\033[1;32mKrivex setup complete!\033[0m"
echo -e "\033[1;32m=========================================================\033[0m"
echo -e "\033[1;33mQuick Start:\033[0m"
echo -e "\033[1;33m  Data pipeline:  ./scripts/data-forge/run_data_forge.sh --dry-run\033[0m"
echo -e "\033[1;33m  Training:       ./scripts/training/train.sh --list\033[0m"
echo -e "\033[1;33m  Inference:      ./scripts/inference/run_inference.sh\033[0m"

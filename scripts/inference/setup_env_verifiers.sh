#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/inference/setup_env_verifiers.sh

echo -e "\033[1;36mInstalling verifier-stack dependencies (CLIP/SigLIP, OCR, etc.)...\033[0m"

if [ ! -d ".venv" ]; then
    echo "Run ./scripts/inference/setup_env.sh first to create .venv"
    exit 1
fi

source .venv/bin/activate
pip install -r inference/requirements-verifiers.txt

echo -e "\033[1;32mDone.\033[0m"

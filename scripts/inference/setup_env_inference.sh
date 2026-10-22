#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/inference/setup_env_inference.sh

echo -e "\033[1;36mInstalling unified inference-layer dependencies into the active venv...\033[0m"
echo -e "\033[1;33mThis includes Planner (Qwen3.5-9B) + Sketch + Polish (Qwen-Image-2.1) + Verifiers.\033[0m"

if [ ! -d ".venv" ]; then
    echo "Run ./scripts/inference/setup_env.sh first to create .venv"
    exit 1
fi

source .venv/bin/activate
pip install -r inference/requirements.txt

echo -e "\033[1;32mDone.\033[0m"

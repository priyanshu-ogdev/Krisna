#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/inference/setup_env_inference.sh

echo -e "\033[1;36mInstalling inference-layer dependencies into the MAIN venv...\033[0m"
echo -e "\033[1;33mThis is Planner (Qwen3.5) + Sketch + both Polish tiers.\033[0m"
echo -e "\033[1;33mThe Critic tier is isolated — see ./scripts/training/setup_env_critic.sh\033[0m"

if [ ! -d ".venv" ]; then
    echo "Run ./scripts/inference/setup_env.sh first to create .venv"
    exit 1
fi

source .venv/bin/activate
pip install -r inference/requirements-inference.txt

echo -e "\033[1;32mDone.\033[0m"

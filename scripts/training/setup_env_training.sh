#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[1;36mInstalling sketch-training dependencies into the MAIN venv...\033[0m"

if [ ! -d ".venv" ]; then
    echo "Run ./scripts/inference/setup_env.sh first to create .venv"
    exit 1
fi

source .venv/bin/activate
pip install -r training/requirements-training.txt

echo -e "\033[1;32mDone.\033[0m"
echo -e "\033[1;33mNext: ./scripts/training/download_vqgan.sh, then ./scripts/training/prepare_sketch_dataset.sh\033[0m"

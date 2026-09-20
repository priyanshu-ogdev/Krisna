#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[1;31m=========================================================\033[0m"
echo -e "\033[1;31mDEPRECATED — the final no-RLHF-loop PRD revision ships the\033[0m"
echo -e "\033[1;31mCritic (Gemma 4 31B) fully frozen and zero-shot. This script\033[0m"
echo -e "\033[1;31mand krisna_training.critic are kept as reference code only —\033[0m"
echo -e "\033[1;31mrunning it trains a real LoRA adapter, but nothing in the\033[0m"
echo -e "\033[1;31mactive inference stack (critic_backend.py/critic_worker.py)\033[0m"
echo -e "\033[1;31mloads any adapter for the Critic. See\033[0m"
echo -e "\033[1;31mtraining/src/krisna_training/critic/__init__.py and\033[0m"
echo -e "\033[1;31mdocs/review/04_critic.md for the full history.\033[0m"
echo -e "\033[1;31m=========================================================\033[0m"

echo -e "\033[1;36mStarting Critic QLoRA training (venv-critic)...\033[0m"

if [ ! -x "./venv-critic/bin/python" ]; then
    echo "venv-critic not found — run ./scripts/training/setup_env_critic.sh first."
    exit 1
fi

./venv-critic/bin/python -m krisna_training.critic.train_critic_qlora \
    --config training/configs/critic_qlora_train.yaml

echo -e "\033[1;32mDone.\033[0m"
echo -e "\033[1;31mNOTE: KRISNA_CRITIC_LORA_PATH is NOT read anywhere in the active\033[0m"
echo -e "\033[1;31minference stack — it was explicitly removed (see\033[0m"
echo -e "\033[1;31minference/src/krisna_inference/backends/factory.py's own\033[0m"
echo -e "\033[1;31m'REMOVED: KRISNA_CRITIC_LORA_PATH' comment, and\033[0m"
echo -e "\033[1;31mtests/inference/test_inference_factory.py, which verifies it's a\033[0m"
echo -e "\033[1;31mno-op). This checkpoint is reference output only — wiring it back\033[0m"
echo -e "\033[1;31min would require restoring the removed adapter-loading path in\033[0m"
echo -e "\033[1;31mfactory.py/critic_backend.py, not just setting this variable.\033[0m"

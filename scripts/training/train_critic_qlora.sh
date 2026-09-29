#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[1;36mStarting Critic QLoRA training (venv-critic)...\033[0m"

if [ ! -x "./venv-critic/bin/python" ]; then
    echo "venv-critic not found — run ./scripts/training/setup_env_critic.sh first."
    exit 1
fi

./venv-critic/bin/python -m krisna_training.critic.train_critic_qlora \
    --config training/configs/critic_qlora_train.yaml

echo -e "\033[1;32mDone.\033[0m"
echo -e "\033[1;33mUse it: export KRISNA_CRITIC_LORA_PATH=./checkpoints/critic_lora/checkpoint_final\033[0m"

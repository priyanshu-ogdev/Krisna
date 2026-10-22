#!/usr/bin/env bash
set -euo pipefail
# Run from the monorepo root: ./scripts/training/train_polish_fast_preview.sh [config.yaml]

CONFIG="${1:-training/configs/cfm_distill_qwen_image21.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo -e "\033[1;31mError: Config $CONFIG not found.\033[0m"
    exit 1
fi

echo -e "\033[1;32mLaunching Phase 4: Consistency Flow Matching (CFM) Distillation\033[0m"
echo -e "\033[1;34mConfig: $CONFIG\033[0m"
echo -e "\033[1;34mTarget: Tier.POLISH_FAST (6-step real-time preview, ~0.9s on A6000)\033[0m"

python3 scripts/training/compare_cfm_steps.py --config "$CONFIG" --selected-steps 6

echo -e "\033[1;32mDone. Distillation receipt at checkpoints/qwen_image21_cfm_fast/distillation_receipt.json.\033[0m"
echo -e "\033[1;33mUse it: export KRIVEX_POLISH_FAST_LORA_PATH=./checkpoints/qwen_image21_cfm_fast\033[0m"

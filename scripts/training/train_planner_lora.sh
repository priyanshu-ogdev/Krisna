#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[1;36mStarting Planner BF16 LoRA training (Qwen3.5-9B)...\033[0m"
source .venv/bin/activate
python -m krisna_training.planner.train --config training/configs/planner_lora_train.yaml
